"""
MM-IMDB image+text → 23-class genre prediction with CoMM + CCS PID.

Dataset (Kaggle: javierurea/simplified-mm-imdb):
  images.npz  — key 'images', shape (N, 3, 256, 160) uint8
  data.npy    — shape (N, 3): [sample_idx, genre_binary_vec (23), movie_plot_text]

Multi-label → single-label: rarest-active-genre strategy
Split: 70 % train / 10 % val / 20 % test

Image encoders (--img-encoder):  resnet50 | convnext (default) | blip_vit
Text  encoders (--txt-encoder):  bert | roberta (default) | deberta
Fusion: CoMM FusionTransformer — CLS + concat(img_tokens, txt_tokens) → 1-layer self-attn
PID:   CCS + Source redundancy → R, U_image, U_text, S  (global + pointwise)

Section 11: top-5 qualitative examples per PID component.
"""

from __future__ import print_function
import argparse
import ast
import copy
import multiprocessing
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils_ours import return_redundancy_test_performances

multiprocessing.set_start_method('fork', force=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
softmax = nn.Softmax(dim=-1)

# Standard MM-IMDB 23 genres (alphabetical encoding order)
GENRE_NAMES = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
    'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Short', 'Sport',
    'Thriller', 'War', 'Western', 'N/A',
]
NUM_GENRES = len(GENRE_NAMES)


# ─── Argument parser ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data-root",       default="/home/rlouiset/imdb")
parser.add_argument("--img-size",        default=224,   type=int)
parser.add_argument("--max-txt-len",     default=512,   type=int)
parser.add_argument("--img-encoder",     default="convnext",
                    choices=["resnet50", "convnext", "blip_vit"])
parser.add_argument("--txt-encoder",     default="roberta",
                    choices=["bert", "roberta", "deberta"])
parser.add_argument("--freeze-encoders", action="store_true")
parser.add_argument("--embed-dim",       default=512,   type=int)
parser.add_argument("--bs",             default=32,    type=int)
parser.add_argument("--num-workers",    default=4,     type=int)
parser.add_argument("--epochs",         default=20,    type=int)
parser.add_argument("--lr",             default=1e-4,  type=float)
parser.add_argument("--weight-decay",   default=1e-2,  type=float)
parser.add_argument("--saved-model",    default=None,
                    help="Save best model to this path during training")
parser.add_argument("--load-model",     default="/home/rlouiset/imdb_model.pth",
                    help="Skip training and load from this path if the file exists")
parser.add_argument("--mod0",           default="image")
parser.add_argument("--mod1",           default="text")
parser.add_argument("--results-dir",    default="/home/rlouiset/imdb/results",
                    help="Directory where top-5 qualitative examples are saved")
args = parser.parse_args()

NUM_CLASSES = NUM_GENRES


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_imdb_data(data_root):
    """Return (images uint8 NCHW, texts list[str], labels_single int64 array)."""
    print("Loading images.npz …")
    images_np = np.load(os.path.join(data_root, "images.npz"))["images"]  # (N,3,H,W) uint8
    print(f"  images : {images_np.shape}  {images_np.dtype}")

    print("Loading data.npy …")
    raw = np.load(os.path.join(data_root, "data.npy"), allow_pickle=True)

    texts, label_vecs = [], []
    for row in raw:
        texts.append(str(row[2]))
        gvec = row[1]
        if isinstance(gvec, str):
            gvec = ast.literal_eval(gvec)
        label_vecs.append(list(gvec))

    label_vecs  = np.array(label_vecs, dtype=np.int32)   # (N, 23)
    genre_freqs = label_vecs.sum(axis=0)                  # (23,)

    print("  Genre frequencies:")
    for i, (name, cnt) in enumerate(zip(GENRE_NAMES, genre_freqs)):
        print(f"    [{i:2d}] {name:<14s}: {cnt}")

    # Multi-label → single-label: rarest active genre per sample
    labels_single = []
    for vec in label_vecs:
        active = np.where(vec > 0)[0]
        if len(active) == 0:
            labels_single.append(NUM_GENRES - 1)           # 'N/A'
        else:
            labels_single.append(int(active[np.argmin(genre_freqs[active])]))
    labels_single = np.array(labels_single, dtype=np.int64)

    single_counts = np.bincount(labels_single, minlength=NUM_GENRES)
    print("  Single-label counts after rarest-genre strategy:")
    for i, c in enumerate(single_counts):
        if c > 0:
            print(f"    {GENRE_NAMES[i]:<14s}: {c}")

    return images_np, texts, labels_single


class MMIMDBDataset(Dataset):
    def __init__(self, images_np, texts, labels, indices, tokenizer,
                 max_txt_len=128, img_size=224, augment=False):
        self.images_np   = images_np
        self.texts       = texts
        self.labels      = labels
        self.indices     = indices
        self.tokenizer   = tokenizer
        self.max_txt_len = max_txt_len

        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if augment:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), norm,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 256 / 224)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(), norm,
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        img = Image.fromarray(self.images_np[ri].transpose(1, 2, 0))  # CHW → HWC
        img = self.tf(img)

        enc = self.tokenizer(
            self.texts[ri],
            max_length=self.max_txt_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return (img,
                enc['input_ids'].squeeze(0),
                enc['attention_mask'].squeeze(0),
                int(self.labels[ri]))


def imdb_collate(batch):
    imgs      = torch.stack([b[0] for b in batch])
    input_ids = torch.stack([b[1] for b in batch])
    attn_mask = torch.stack([b[2] for b in batch])
    labels    = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return imgs, input_ids, attn_mask, labels


def get_dataloaders(images_np, texts, labels_single, tokenizer,
                    bs, num_workers, img_size=224, max_txt_len=128):
    n     = len(labels_single)
    perm  = np.random.default_rng(42).permutation(n)
    n_tst = int(n * 0.20)
    n_val = int(n * 0.10)
    test_idx  = perm[:n_tst].tolist()
    val_idx   = perm[n_tst:n_tst + n_val].tolist()
    train_idx = perm[n_tst + n_val:].tolist()

    kw = dict(tokenizer=tokenizer, max_txt_len=max_txt_len, img_size=img_size)
    train_ds = MMIMDBDataset(images_np, texts, labels_single, train_idx, augment=True,  **kw)
    val_ds   = MMIMDBDataset(images_np, texts, labels_single, val_idx,   augment=False, **kw)
    test_ds  = MMIMDBDataset(images_np, texts, labels_single, test_idx,  augment=False, **kw)

    ldr = dict(batch_size=bs, num_workers=num_workers,
               collate_fn=imdb_collate, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **ldr),
        DataLoader(val_ds,   shuffle=False, **ldr),
        DataLoader(test_ds,  shuffle=False, **ldr),
        train_idx, val_idx, test_idx,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODERS  (module-level classes required for torch.save / pickle)
# ═══════════════════════════════════════════════════════════════════════════════

class BlipVitEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(pixel_values=x).last_hidden_state


class HFTextEncoder(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state


def build_img_encoder(name):
    if name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        enc = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                            m.layer1, m.layer2, m.layer3, m.layer4)
        return enc, 2048, True
    if name == "convnext":
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        m = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        return m.features, 1024, True
    if name == "blip_vit":
        from transformers import BlipVisionModel
        m = BlipVisionModel.from_pretrained("Salesforce/blip-image-captioning-base")
        return BlipVitEncoder(m), 768, False
    raise ValueError(name)


def build_txt_encoder(name):
    hub = {"bert":    "bert-base-uncased",
           "roberta": "roberta-base",
           "deberta": "microsoft/deberta-v3-base"}
    from transformers import AutoModel
    return HFTextEncoder(AutoModel.from_pretrained(hub[name])), 768


def build_tokenizer(name):
    hub = {"bert":    "bert-base-uncased",
           "roberta": "roberta-base",
           "deberta": "microsoft/deberta-v3-base"}
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hub[name])


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN ADAPTERS + FUSION TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_2d_sincos_posemb(h, w, embed_dim):
    assert embed_dim % 4 == 0
    pos_dim = embed_dim // 4
    omega   = 1. / (10000 ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
    grid_w, grid_h = torch.meshgrid(
        torch.arange(w, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32),
        indexing='ij',
    )
    out_w = torch.einsum('m,d->md', grid_w.reshape(-1), omega)
    out_h = torch.einsum('m,d->md', grid_h.reshape(-1), omega)
    pos   = torch.cat([torch.sin(out_w), torch.cos(out_w),
                       torch.sin(out_h), torch.cos(out_h)], dim=1)
    return pos.reshape(h, w, embed_dim).permute(2, 0, 1).unsqueeze(0)


class SpatialTokenAdapter(nn.Module):
    """(B, C, H, W) → (B, H*W, embed_dim) with 2D sincos positional embedding."""
    def __init__(self, in_channels, embed_dim, image_size=7):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.register_buffer('pos_emb',
                             _build_2d_sincos_posemb(image_size, image_size, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = self.proj(x)
        pos    = F.interpolate(self.pos_emb, size=(H, W), mode='bicubic', align_corners=False)
        return (tokens + pos).flatten(2).transpose(1, 2)


class SequenceTokenAdapter(nn.Module):
    """(B, T, in_dim) → (B, T, embed_dim)."""
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class FusionTransformer(nn.Module):
    """CLS + concat(img_tokens, txt_tokens) → 1-layer pre-norm self-attention → CLS."""
    def __init__(self, width=512, n_heads=8, n_layers=1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, sequences):
        B = sequences[0].size(0)
        x = torch.cat([self.cls_token.expand(B, -1, -1), torch.cat(sequences, dim=1)], dim=1)
        return self.norm(self.transformer(x)[:, 0])


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ImdbCoMMModel(nn.Module):
    def __init__(self, num_classes=NUM_GENRES, img_encoder_type="convnext",
                 txt_encoder_type="roberta", embed_dim=512, freeze_encoders=False):
        super().__init__()

        img_enc, img_dim, img_spatial = build_img_encoder(img_encoder_type)
        txt_enc, txt_dim              = build_txt_encoder(txt_encoder_type)

        self.img_encoder = img_enc
        self.txt_encoder = txt_enc
        self.img_spatial = img_spatial

        self.img_adapter = (SpatialTokenAdapter(img_dim, embed_dim, image_size=7)
                            if img_spatial else SequenceTokenAdapter(img_dim, embed_dim))
        self.txt_adapter = SequenceTokenAdapter(txt_dim, embed_dim)
        self.fusion      = FusionTransformer(width=embed_dim, n_heads=8, n_layers=1)
        self.head        = nn.Linear(embed_dim, num_classes)
        self.mod_heads   = nn.ModuleList([nn.Linear(embed_dim, num_classes),
                                          nn.Linear(embed_dim, num_classes)])
        self.reps = []   # mean-pooled (B, D) per modality — populated in forward

        if freeze_encoders:
            for p in self.img_encoder.parameters(): p.requires_grad = False
            for p in self.txt_encoder.parameters(): p.requires_grad = False

    def forward(self, imgs, input_ids, attention_mask):
        img_tokens = self.img_adapter(self.img_encoder(imgs))
        txt_tokens = self.txt_adapter(self.txt_encoder(input_ids, attention_mask))

        self.reps  = [img_tokens.mean(dim=1), txt_tokens.mean(dim=1)]

        fused        = self.fusion([img_tokens, txt_tokens])
        joint_logits = self.head(fused)
        mod_logits   = [h(r) for h, r in zip(self.mod_heads, self.reps)]
        return joint_logits, [img_tokens, txt_tokens], mod_logits


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST / EXTRACT
# ═══════════════════════════════════════════════════════════════════════════════

def train_comm(model, traindata, validdata, epochs, lr, weight_decay, save=None):
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_model = copy.deepcopy(model)
    best_val   = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.; n = 0
        for imgs, inp, msk, y in traindata:
            imgs, inp, msk, y = (imgs.to(device), inp.to(device),
                                 msk.to(device), y.to(device))
            joint, _, mod_logits = model(imgs, inp, msk)
            loss = criterion(joint, y) + sum(criterion(l, y) for l in mod_logits)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()
            total_loss += loss.item() * len(y); n += len(y)
        scheduler.step()

        model.eval(); val_loss = 0.; vn = 0; vc = 0
        with torch.no_grad():
            for imgs, inp, msk, y in validdata:
                imgs, inp, msk, y = (imgs.to(device), inp.to(device),
                                     msk.to(device), y.to(device))
                joint, _, _ = model(imgs, inp, msk)
                val_loss += criterion(joint, y).item() * len(y)
                vn += len(y)
                vc += (joint.argmax(1) == y).sum().item()
        val_loss /= vn
        print(f"Epoch {epoch:02d}  train={total_loss/n:.4f}  "
              f"val={val_loss:.4f}  acc={vc/vn:.4f}")
        if val_loss < best_val:
            best_val = val_loss; best_model = copy.deepcopy(model)
            print("  → best")
            if save:
                torch.save(model, save)

    return best_model


def test_comm(model, testdata):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    pj, pm0, pm1, ys = [], [], [], []
    tj = tm0 = tm1 = n = 0

    with torch.no_grad():
        for imgs, inp, msk, y in testdata:
            imgs, inp, msk, y = (imgs.to(device), inp.to(device),
                                 msk.to(device), y.to(device))
            joint, _, mod_logits = model(imgs, inp, msk)
            tj  += criterion(joint,         y).item() * len(y)
            tm0 += criterion(mod_logits[0], y).item() * len(y)
            tm1 += criterion(mod_logits[1], y).item() * len(y)
            n   += len(y)
            pj.append(joint.cpu());          pm0.append(mod_logits[0].cpu())
            pm1.append(mod_logits[1].cpu()); ys.append(y.cpu())

    pj  = torch.cat(pj);  pm0 = torch.cat(pm0)
    pm1 = torch.cat(pm1); ys  = torch.cat(ys)
    return {
        "joint_acc":       (pj.argmax(1)  == ys).float().mean().item(),
        "modalities_acc": [(pm0.argmax(1) == ys).float().mean().item(),
                           (pm1.argmax(1) == ys).float().mean().item()],
        "joint_ce":        tj  / n,
        "modalities_ce":  [tm0 / n, tm1 / n],
        "pred_joint":      pj,
        "pred_modalities": [pm0, pm1],
        "true_labels":     ys,
    }


def extract_split(model, loader):
    model.eval()
    r0, r1, tgts = [], [], []
    with torch.no_grad():
        for imgs, inp, msk, y in loader:
            _ = model(imgs.to(device), inp.to(device), msk.to(device))
            r0.append(model.reps[0].cpu())
            r1.append(model.reps[1].cpu())
            tgts.append(y.cpu())
    X = {"modality0": torch.cat(r0).float(), "modality1": torch.cat(r1).float()}
    return X, torch.cat(tgts).long()


# ═══════════════════════════════════════════════════════════════════════════════
# PID UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):
    probs = torch.clamp(probs, eps, 1.0)
    ce  = -torch.log(probs)[torch.arange(len(targets)), targets.long()].mean()
    acc = (probs.argmax(1) == targets).float().mean()
    return acc.item(), ce.item()


def compute_log_py(targets, num_classes):
    counts = torch.bincount(targets.long(), minlength=num_classes).float()
    probs  = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return torch.log(probs)[targets.long()]


def ce_per_sample(targets, probs, eps=1e-12):
    return -torch.log(torch.clamp(probs, eps, 1.0))[torch.arange(len(targets)), targets.long()]


def compute_probs_and_ce(logits_list, targets):
    probs_list = [F.softmax(l, dim=1) for l in logits_list]
    ce_list    = [ce_per_sample(targets, p) for p in probs_list]
    return probs_list, ce_list


def compute_entropy_from_targets(targets, num_classes):
    targets = torch.as_tensor(targets).long()
    counts  = torch.bincount(targets, minlength=num_classes).float()
    probs   = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return -(probs * torch.log(probs)).sum().item()


def logp(p):
    return torch.log(torch.clamp(p, 1e-12, 1.0))


def compute_pid_global(joint_ce, mod0_ce, mod1_ce, red_ce, src_red_ce,
                       targets, num_classes, mod0_name="image", mod1_name="text"):
    hy = compute_entropy_from_targets(targets, num_classes)
    print(f"H(Y)={hy:.4f}  joint={joint_ce:.4f}  red={red_ce:.4f}  "
          f"src_red={src_red_ce:.4f}  {mod0_name}={mod0_ce:.4f}  "
          f"{mod1_name}={mod1_ce:.4f}")

    mod0_ce    = min(mod0_ce,    hy); mod1_ce    = min(mod1_ce,    hy)
    red_ce     = min(red_ce,     hy); src_red_ce = min(src_red_ce, hy)
    red_ce     = max(red_ce,     joint_ce, mod0_ce, mod1_ce)
    src_red_ce = max(src_red_ce, joint_ce, mod0_ce, mod1_ce)
    red_ce     = min(red_ce,     src_red_ce)
    mod0_ce    = min(max(mod0_ce, joint_ce), red_ce)
    mod1_ce    = min(max(mod1_ce, joint_ce), red_ce)

    i_total = hy - joint_ce
    i_r     = hy - red_ce;   i_r_src = hy - src_red_ce
    i_u0    = (hy - mod0_ce) - i_r
    i_u1    = (hy - mod1_ce) - i_r
    i_s     = i_total - i_u0 - i_u1 - i_r

    if i_s < 0:
        i_r    -= i_s; i_r_src -= i_s
        i_u0 = (hy - mod0_ce) - i_r
        i_u1 = (hy - mod1_ce) - i_r
        i_s  = 0.0

    ratio = i_r_src / (i_r + 1e-10)
    print(f"R={i_r:.4f} ({100*ratio:.1f}% Source)  "
          f"U_{mod0_name}={i_u0:.4f}  U_{mod1_name}={i_u1:.4f}  "
          f"S={i_s:.4f}  I_total={i_total:.4f}")


def compute_redundancy_metrics(y_pred_dict):
    results = {}
    for key in ["modality0", "modality1", "average"]:
        acc, ce = traditional_cross_entropy_from_probs(
            softmax(y_pred_dict[key]), y_pred_dict["targets"])
        results[key] = {"accuracy": acc, "cross_entropy": ce}
    return results


def compute_pointwise_pid_with_source(d, num_classes):
    targets = d["true_labels"].long()
    log_py  = compute_log_py(targets, num_classes)
    pid_list = []

    for j, m0, m1, npr, r_src, y, lpy in zip(
        d["pred_joint"],
        d["pred_modalities"][0],
        d["pred_modalities"][1],
        d["redundancy_pointwise_ce"],
        d["source_redundancy_preds"],
        targets,
        log_py,
    ):
        y        = y.long()
        joint_ce = -logp(F.softmax(j,     dim=0))[y]
        mod0_ce  = -logp(F.softmax(m0,    dim=0))[y]
        mod1_ce  = -logp(F.softmax(m1,    dim=0))[y]
        red_ce   = float(npr)
        src_ce   = -logp(F.softmax(r_src, dim=0))[y]
        hy       = float(-lpy)

        joint_ce = min(red_ce, float(joint_ce))
        src_ce   = min(red_ce, float(src_ce))
        red_ce   = min(red_ce, hy)
        src_ce   = min(src_ce, hy)
        mod0_ce  = max(float(mod0_ce), joint_ce)
        mod1_ce  = max(float(mod1_ce), joint_ce)

        total = hy - joint_ce
        r_val = max(hy - red_ce, hy - src_ce)
        u0    = hy - mod0_ce - r_val
        u1    = hy - mod1_ce - r_val
        s     = total - u0 - u1 - r_val
        pid_list.append([u0, u1, r_val, s])

    return np.array(pid_list, dtype=float)


def normalize_pid(pid):
    pid_ = np.maximum(pid, 0)
    pid_ /= pid_.sum(axis=1, keepdims=True) + 1e-12
    return pid_


def print_model_metrics(d):
    print(f"{'Model':<12s} | {'Acc':>8s} | {'CE':>8s}")
    print("-" * 36)
    print(f"{'Joint':<12s} | {d['joint_acc']:8.4f} | {d['joint_ce']:8.4f}")
    print(f"{'Image':<12s} | {d['modalities_acc'][0]:8.4f} | {d['modalities_ce'][0]:8.4f}")
    print(f"{'Text':<12s}  | {d['modalities_acc'][1]:8.4f} | {d['modalities_ce'][1]:8.4f}")


def print_redundancy_metrics(results):
    for key, name in [("modality0", "Red Image"), ("modality1", "Red Text"),
                      ("average",   "Red Joint")]:
        print(f"{name:<14s} | {results[key]['accuracy']:8.4f} | "
              f"{results[key]['cross_entropy']:8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"MM-IMDB CoMM CCS-PID  ({NUM_GENRES} genres)")
    print(f"  Image encoder : {args.img_encoder}  (img_size={args.img_size})")
    print(f"  Text  encoder : {args.txt_encoder}  (max_len={args.max_txt_len})")
    print(f"  Embed dim     : {args.embed_dim}")
    print(f"  Device        : {device}")
    print(f"{'='*60}\n")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer…")
    tokenizer = build_tokenizer(args.txt_encoder)

    # ── 2. Data ───────────────────────────────────────────────────────────────
    print("\nLoading data…")
    images_np, texts, labels_single = load_imdb_data(args.data_root)

    traindata, valdata, testdata, train_idx, val_idx, test_idx = get_dataloaders(
        images_np, texts, labels_single, tokenizer,
        bs=args.bs, num_workers=args.num_workers,
        img_size=args.img_size, max_txt_len=args.max_txt_len,
    )
    test_texts  = [texts[i]          for i in test_idx]   # raw strings for display
    test_labels = labels_single[test_idx]                  # int64 (N_test,)
    print(f"\n  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("\nBuilding model…")
    model = ImdbCoMMModel(
        num_classes=NUM_CLASSES,
        img_encoder_type=args.img_encoder,
        txt_encoder_type=args.txt_encoder,
        embed_dim=args.embed_dim,
        freeze_encoders=args.freeze_encoders,
    ).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,}  (trainable: {trainable:,})")

    # ── 4. Train or load ──────────────────────────────────────────────────────
    if args.load_model and os.path.isfile(args.load_model):
        print(f"\nLoading model from {args.load_model}…")
        model = torch.load(args.load_model, map_location=device, weights_only=False)
        model = model.to(device)
        print("  Model loaded — skipping training.")
    else:
        print("\nTraining…")
        model = train_comm(model, traindata, valdata,
                           epochs=args.epochs, lr=args.lr,
                           weight_decay=args.weight_decay,
                           save=args.saved_model)

    # ── 5. Test ───────────────────────────────────────────────────────────────
    print("\nTest performance:")
    d = test_comm(model, testdata)
    print_model_metrics(d)

    # ── 6. Extract representations ────────────────────────────────────────────
    print("\nExtracting representations…")
    X_train, y_train = extract_split(model, traindata)
    X_val,   y_val   = extract_split(model, valdata)
    X_test,  y_test  = extract_split(model, testdata)
    print(f"  embed_dim={X_train['modality0'].shape[1]}  "
          f"train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    # ── 7. CCS redundancy ─────────────────────────────────────────────────────
    targets = d["true_labels"].long()
    log_py  = compute_log_py(targets, NUM_CLASSES)
    _, ce_list = compute_probs_and_ce(d["pred_modalities"], targets)

    i_list    = [-ce - log_py for ce in ce_list]
    same_sign = torch.sign(i_list[0]) == torch.sign(i_list[1])
    worst_ce  = torch.max(torch.stack(ce_list, dim=1), dim=1).values
    ccs       = torch.where(same_sign, worst_ce, -log_py)

    d["redundancy_ce"]           = ccs.mean().item()
    d["redundancy_pointwise_ce"] = ccs.numpy()

    # ── 8. Source redundancy ──────────────────────────────────────────────────
    print("\nFitting source redundancy model…")
    y_pred_dict = return_redundancy_test_performances(
        X_train, X_val, X_test,
        y_train.long(), y_val.long(), y_test.long(),
        f"imdb_{args.img_encoder}_{args.txt_encoder}",
        distribution_target="categorical",
        num_classes=NUM_CLASSES,
    )
    results = compute_redundancy_metrics(y_pred_dict)
    print_redundancy_metrics(results)

    d["source_redundancy_pointwise_ce"] = results["average"]["cross_entropy"]
    d["source_redundancy_preds"]        = y_pred_dict["average"]

    # ── 9. Global PID ─────────────────────────────────────────────────────────
    print("\nGlobal PID:")
    compute_pid_global(
        d["joint_ce"],
        d["modalities_ce"][0],
        d["modalities_ce"][1],
        d["redundancy_ce"],
        d["source_redundancy_pointwise_ce"],
        targets=d["true_labels"],
        num_classes=NUM_CLASSES,
        mod0_name=args.mod0,
        mod1_name=args.mod1,
    )

    # ── 10. Pointwise PID + correction loop ───────────────────────────────────
    pid_source = compute_pointwise_pid_with_source(d, NUM_CLASSES)
    print(f"\nMean pointwise PID  [U_{args.mod0}, U_{args.mod1}, R, S]:")
    print(np.mean(pid_source, axis=0))
    print("Normalised mean:", np.mean(normalize_pid(pid_source), axis=0))

    for i, pid_i in enumerate(pid_source):
        if pid_i[0] < 0 and pid_i[1] >= 0:
            pid_source[i] = [0, pid_i[1], pid_i[2], pid_i[3] + pid_i[0]]
        if pid_i[1] < 0 and pid_i[0] >= 0:
            pid_source[i] = [pid_i[0], 0, pid_i[2], pid_i[3] + pid_i[1]]

    print(f"\nAfter correction  Mean pointwise PID  [U_{args.mod0}, U_{args.mod1}, R, S]:")
    print(np.mean(pid_source, axis=0))
    pid_norm = normalize_pid(pid_source)
    print("After correction  Normalised mean:", np.mean(pid_norm, axis=0))

    # ── 11. Save top-5 qualitative examples per PID component ────────────────
    # pid_norm columns: 0 = U_img, 1 = U_txt, 2 = R, 3 = S
    comp_dirs = [f"U_{args.mod0}", f"U_{args.mod1}", "R", "S"]

    for col, cdir in enumerate(comp_dirs):
        out_dir = os.path.join(args.results_dir, cdir)
        os.makedirs(out_dir, exist_ok=True)
        top5 = np.argsort(pid_norm[:, col])[-5:][::-1]
        for rank, i in enumerate(top5, 1):
            real_i  = test_idx[i]
            pv      = pid_norm[i]
            genre   = GENRE_NAMES[test_labels[i]]

            # save image
            img_pil = Image.fromarray(images_np[real_i].transpose(1, 2, 0))
            img_pil.save(os.path.join(out_dir, f"rank{rank}.png"))

            # save text + metadata
            with open(os.path.join(out_dir, f"rank{rank}_info.txt"), 'w') as f:
                f.write(f"Rank    : {rank}\n")
                f.write(f"Genre   : {genre}\n")
                f.write(f"PID %   : U_img={100*pv[0]:.1f}%  "
                        f"U_txt={100*pv[1]:.1f}%  "
                        f"R={100*pv[2]:.1f}%  "
                        f"S={100*pv[3]:.1f}%\n")
                f.write(f"\nText:\n{test_texts[i]}\n")

    print(f"\nTop-5 examples saved to {args.results_dir}/")
