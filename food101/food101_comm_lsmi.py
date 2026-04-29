"""
Food101 image-text CoMM + LSMI pointwise PID estimator.

Pipeline:
  1. Train CoMM model (image encoder → spatial tokens + text encoder → tokens
     → FusionTransformer CLS)
  2. Extract mean-pooled representations (embed_dim-d per modality)
  3. Train LSMI discriminators p(y|z_img), p(y|z_txt), p(y|z_img,z_txt)
  4. Train MargKernel entropy estimators H(Z_img), H(Z_txt)
  5. Compute pointwise LSMI PID → R, U_img, U_txt, S

Image encoders (--img-encoder): resnet50 | convnext | blip_vit
Text  encoders (--txt-encoder): bert     | roberta  | deberta
"""

from __future__ import print_function
import argparse
import copy
import csv
import html
import math
import multiprocessing
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from utils_lsmi import MargKernel, cls_network, feature_dataset, setup_seed

multiprocessing.set_start_method('fork', force=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data-root",   default="/home/rlouiset/Food101", type=str)
parser.add_argument("--img-size",    default=224, type=int)
parser.add_argument("--max-txt-len", default=64,  type=int)
parser.add_argument("--val-split",   default=0.1, type=float)
# Modality names (used only for display)
parser.add_argument("--mod0",        default="image", type=str)
parser.add_argument("--mod1",        default="text",  type=str)
# Encoders
parser.add_argument("--img-encoder", default="convnext",
                    choices=["resnet50", "convnext", "blip_vit"])
parser.add_argument("--txt-encoder", default="roberta",
                    choices=["bert", "roberta", "deberta"])
parser.add_argument("--freeze-encoders", action="store_true")
# Model training
parser.add_argument("--embed-dim",   default=512, type=int)
parser.add_argument("--bs",          default=32,  type=int)
parser.add_argument("--num-workers", default=4,   type=int)
parser.add_argument("--epochs",      default=20,  type=int)
parser.add_argument("--lr",          default=1e-4, type=float)
parser.add_argument("--weight-decay",default=1e-2, type=float)
parser.add_argument("--saved-model", default="/home/rlouiset/food101_model.pth", type=str)
# LSMI
parser.add_argument("--lsmi-embed-size",     default=128, type=int,
                    help="Hidden dim for LSMI discriminators")
parser.add_argument("--lsmi-bs",             default=512, type=int)
parser.add_argument("--epochs-discriminator",default=30,  type=int)
parser.add_argument("--epochs-entropy",      default=30,  type=int)
parser.add_argument("--seed",                default=1,   type=int)
args = parser.parse_args()

setup_seed(args.seed)
NUM_CLASSES = 101


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class Food101Dataset(Dataset):
    """CSV format (no header): filename, web_title, class_label"""

    def __init__(self, csv_path, img_root, tokenizer, transform=None,
                 max_txt_len=64, class_to_idx=None):
        self.img_root    = img_root
        self.transform   = transform
        self.tokenizer   = tokenizer
        self.max_txt_len = max_txt_len

        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                filename = row[0].strip()
                title    = html.unescape(row[1].strip())
                cls      = row[2].strip()
                self.samples.append((filename, title, cls))

        if class_to_idx is None:
            classes = sorted({s[2] for s in self.samples})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, title, cls = self.samples[idx]
        img = Image.open(os.path.join(self.img_root, cls, filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        enc = self.tokenizer(title, max_length=self.max_txt_len,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        return (img,
                enc['input_ids'].squeeze(0),
                enc['attention_mask'].squeeze(0),
                self.class_to_idx[cls])


def food101_collate(batch):
    imgs      = torch.stack([b[0] for b in batch])
    input_ids = torch.stack([b[1] for b in batch])
    attn_mask = torch.stack([b[2] for b in batch])
    labels    = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return imgs, input_ids, attn_mask, labels


def get_dataloaders(data_root, tokenizer, bs, num_workers, img_size=224,
                    max_txt_len=64, val_split=0.1):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), normalize,
    ])

    img_train_root = os.path.join(data_root, "images", "train")
    img_test_root  = os.path.join(data_root, "images", "test")
    txt_train_csv  = os.path.join(data_root, "texts", "train_titles.csv")
    txt_test_csv   = os.path.join(data_root, "texts", "test_titles.csv")

    ds_aug   = Food101Dataset(txt_train_csv, img_train_root, tokenizer, train_tf, max_txt_len)
    ds_noaug = Food101Dataset(txt_train_csv, img_train_root, tokenizer, test_tf,  max_txt_len,
                              class_to_idx=ds_aug.class_to_idx)
    class_to_idx = ds_aug.class_to_idx

    n      = len(ds_aug)
    n_val  = int(n * val_split)
    perm   = np.random.default_rng(42).permutation(n)
    tr_idx = perm[n_val:].tolist()
    va_idx = perm[:n_val].tolist()

    train_ds       = Subset(ds_aug,   tr_idx)
    val_ds         = Subset(ds_noaug, va_idx)
    train_clean_ds = Subset(ds_noaug, tr_idx)   # no augmentation, used for LSMI
    test_ds        = Food101Dataset(txt_test_csv, img_test_root, tokenizer, test_tf,
                                    max_txt_len, class_to_idx=class_to_idx)

    loader_kw = dict(batch_size=bs, num_workers=num_workers,
                     collate_fn=food101_collate, pin_memory=True)
    return (
        DataLoader(train_ds,       shuffle=True,  **loader_kw),
        DataLoader(val_ds,         shuffle=False, **loader_kw),
        DataLoader(test_ds,        shuffle=False, **loader_kw),
        DataLoader(train_clean_ds, shuffle=False, **loader_kw),
        class_to_idx,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODERS
# ═══════════════════════════════════════════════════════════════════════════════

class BlipVitEncoder(nn.Module):
    """Wraps BlipVisionModel → last_hidden_state (B, T, 768)."""
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(pixel_values=x).last_hidden_state


class HFTextEncoder(nn.Module):
    """Wraps a HuggingFace AutoModel → last_hidden_state (B, T, D)."""
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
        encoder = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                m.layer1, m.layer2, m.layer3, m.layer4)
        return encoder, 2048, True
    elif name == "convnext":
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        m = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        return m.features, 1024, True
    elif name == "blip_vit":
        from transformers import BlipVisionModel
        m = BlipVisionModel.from_pretrained("Salesforce/blip-image-captioning-base")
        return BlipVitEncoder(m), 768, False
    raise ValueError(name)


def build_txt_encoder(name):
    hub = {"bert": "bert-base-uncased",
           "roberta": "roberta-base",
           "deberta": "microsoft/deberta-v3-base"}
    from transformers import AutoModel
    return HFTextEncoder(AutoModel.from_pretrained(hub[name])), 768


def build_tokenizer(name):
    hub = {"bert": "bert-base-uncased",
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
    grid_w, grid_h = torch.meshgrid(torch.arange(w, dtype=torch.float32),
                                     torch.arange(h, dtype=torch.float32),
                                     indexing='ij')
    out_w = torch.einsum('m,d->md', grid_w.reshape(-1), omega)
    out_h = torch.einsum('m,d->md', grid_h.reshape(-1), omega)
    pos   = torch.cat([torch.sin(out_w), torch.cos(out_w),
                       torch.sin(out_h), torch.cos(out_h)], dim=1)
    return pos.reshape(h, w, embed_dim).permute(2, 0, 1).unsqueeze(0)


class SpatialTokenAdapter(nn.Module):
    def __init__(self, in_channels, embed_dim, image_size=7):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        pos = _build_2d_sincos_posemb(image_size, image_size, embed_dim)
        self.register_buffer('pos_emb', pos)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = self.proj(x)
        pos    = F.interpolate(self.pos_emb, size=(H, W),
                               mode='bicubic', align_corners=False)
        return (tokens + pos).flatten(2).transpose(1, 2)


class SequenceTokenAdapter(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class FusionTransformer(nn.Module):
    def __init__(self, width=512, n_heads=8, n_layers=1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, sequences):
        B   = sequences[0].size(0)
        x   = torch.cat(sequences, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        return self.norm(self.transformer(x)[:, 0])


class Food101CoMMModel(nn.Module):
    def __init__(self, num_classes=101, img_encoder_type="convnext",
                 txt_encoder_type="roberta", embed_dim=512, freeze_encoders=False):
        super().__init__()
        img_enc, img_dim, img_spatial = build_img_encoder(img_encoder_type)
        txt_enc, txt_dim              = build_txt_encoder(txt_encoder_type)
        self.img_encoder = img_enc
        self.txt_encoder = txt_enc
        self.img_spatial = img_spatial
        self.img_adapter = (SpatialTokenAdapter(img_dim, embed_dim)
                            if img_spatial else SequenceTokenAdapter(img_dim, embed_dim))
        self.txt_adapter = SequenceTokenAdapter(txt_dim, embed_dim)
        self.fusion    = FusionTransformer(width=embed_dim, n_heads=8, n_layers=1)
        self.head      = nn.Linear(embed_dim, num_classes)
        self.mod_heads = nn.ModuleList([nn.Linear(embed_dim, num_classes),
                                        nn.Linear(embed_dim, num_classes)])
        self.reps = []   # mean-pooled (B, embed_dim) per modality

        if freeze_encoders:
            for p in self.img_encoder.parameters(): p.requires_grad = False
            for p in self.txt_encoder.parameters(): p.requires_grad = False

    def forward(self, imgs, input_ids, attention_mask):
        img_feat   = self.img_encoder(imgs)
        img_tokens = self.img_adapter(img_feat)
        txt_tokens = self.txt_adapter(self.txt_encoder(input_ids, attention_mask))
        self.reps  = [img_tokens.mean(dim=1), txt_tokens.mean(dim=1)]
        fused        = self.fusion([img_tokens, txt_tokens])
        joint_logits = self.head(fused)
        mod_logits   = [h(r) for h, r in zip(self.mod_heads, self.reps)]
        return joint_logits, [img_tokens, txt_tokens], mod_logits


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def train_comm(model, traindata, validdata, epochs, lr, weight_decay, save=None):
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_model = copy.deepcopy(model)
    best_val   = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; n = 0
        for imgs, inp, msk, y in traindata:
            imgs, inp, msk, y = (imgs.to(device), inp.to(device),
                                 msk.to(device), y.to(device).long())
            joint, _, mod_logits = model(imgs, inp, msk)
            loss = criterion(joint, y) + sum(criterion(l, y) for l in mod_logits)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()
            total_loss += loss.item() * len(y); n += len(y)
        scheduler.step()

        model.eval()
        val_loss = 0.0; vn = 0; vc = 0
        with torch.no_grad():
            for imgs, inp, msk, y in validdata:
                imgs, inp, msk, y = (imgs.to(device), inp.to(device),
                                     msk.to(device), y.to(device).long())
                joint, _, _ = model(imgs, inp, msk)
                val_loss += criterion(joint, y).item() * len(y)
                vn += len(y)
                vc += (joint.argmax(1) == y).sum().item()
        val_loss /= vn
        print(f"Epoch {epoch:02d}  train={total_loss/n:.4f}  val={val_loss:.4f}  acc={vc/vn:.4f}")
        if val_loss < best_val:
            best_val   = val_loss
            best_model = copy.deepcopy(model)
            print("  -> Best")
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
                                 msk.to(device), y.to(device).long())
            joint, _, mod_logits = model(imgs, inp, msk)
            tj  += criterion(joint,         y).item() * len(y)
            tm0 += criterion(mod_logits[0], y).item() * len(y)
            tm1 += criterion(mod_logits[1], y).item() * len(y)
            n   += len(y)
            pj.append(joint.cpu()); pm0.append(mod_logits[0].cpu())
            pm1.append(mod_logits[1].cpu()); ys.append(y.cpu())

    pj  = torch.cat(pj);  pm0 = torch.cat(pm0)
    pm1 = torch.cat(pm1); ys  = torch.cat(ys)

    def acc(t, p): return (p.argmax(1) == t).float().mean().item()
    print(f"{'Model':<12s} | {'Acc':>8s} | {'CE':>8s}")
    print("-" * 36)
    print(f"{'Joint':<12s} | {acc(ys, pj):8.4f} | {tj/n:8.4f}")
    print(f"{args.mod0:<12s} | {acc(ys, pm0):8.4f} | {tm0/n:8.4f}")
    print(f"{args.mod1:<12s} | {acc(ys, pm1):8.4f} | {tm1/n:8.4f}")


# ─── Representation extraction ────────────────────────────────────────────────

def extract_representations(model, loader):
    """Returns (N, embed_dim) tensors for each modality and the labels."""
    model.eval()
    reps0, reps1, targets = [], [], []
    with torch.no_grad():
        for imgs, inp, msk, y in loader:
            _ = model(imgs.to(device), inp.to(device), msk.to(device))
            reps0.append(model.reps[0].cpu())   # already mean-pooled (B, D)
            reps1.append(model.reps[1].cpu())
            targets.append(y.cpu())
    return (torch.cat(reps0).float(),
            torch.cat(reps1).float(),
            torch.cat(targets).long())


# ═══════════════════════════════════════════════════════════════════════════════
# LSMI ESTIMATORS
# ═══════════════════════════════════════════════════════════════════════════════

def obtain_discriminator(train_loader, input_size_1, input_size_2,
                         embed_size, n_classes, n_epochs):
    m1 = cls_network(input_size_1,                embed_size, n_classes).to(device)
    m2 = cls_network(input_size_2,                embed_size, n_classes).to(device)
    mj = cls_network(input_size_1 + input_size_2, embed_size, n_classes).to(device)
    models    = [m1, m2, mj]
    optimizer = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        losses = 0.0; num_samples = 0
        for batch in train_loader:
            x1, x2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            loss = (criterion(m1(x1), y) + criterion(m2(x2), y)
                    + criterion(mj(torch.cat([x1, x2], dim=1)), y))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses += loss.item() * len(y); num_samples += len(y)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  [disc]    epoch {epoch+1}/{n_epochs}  loss={losses/num_samples:.4f}")
    return models


def obtain_entropy_estimator(train_loader, input_size_1, input_size_2, n_epochs):
    m1 = MargKernel(input_size_1).to(device)
    m2 = MargKernel(input_size_2).to(device)
    models    = [m1, m2]
    optimizer = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(n_epochs):
        for m in models: m.train()
        losses = 0.0
        for batch in train_loader:
            x1, x2 = batch[0].to(device), batch[1].to(device)
            loss = m1(x1) + m2(x2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses += loss.item() * len(x1)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  [entropy] epoch {epoch+1}/{n_epochs}  loss={losses/len(train_loader.dataset):.4f}")
    return models


def get_mutual_info(loader, model, modality, n_classes):
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch[0].to(device)
            x2 = batch[1].to(device)
            y  = batch[2].to(device)
            if modality == 'modality_1':
                x = x1
            elif modality == 'modality_2':
                x = x2
            else:
                x = torch.cat([x1, x2], dim=1)
            rows = torch.arange(x.size(0), device=device)
            out  = model(x)
            info.append(math.log(n_classes) + F.log_softmax(out, dim=1)[rows, y])
    return torch.cat(info).detach()


def get_entropy(loader, model, modality):
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            x = (batch[0] if modality == 'modality_1' else batch[1]).to(device)
            info.append(model(x))
    return torch.cat(info).detach()


def RUS_adjustment(rus):
    r, u1, u2, s = rus
    R_mean  = r.detach().mean()
    U1_mean = u1.detach().mean()
    U2_mean = u2.detach().mean()
    S_mean  = s.detach().mean()
    adj = torch.tensor(0.0, dtype=R_mean.dtype, device=R_mean.device)
    if R_mean < 0 or S_mean < 0:
        adj = -torch.min(R_mean, S_mean)
    elif U1_mean < 0 or U2_mean < 0:
        adj = torch.min(U1_mean, U2_mean)
    return r + adj, u1 - adj, u2 - adj, s + adj


def LSMI_estimation(loader, discriminators, entropy_estimators, n_classes, split_name=""):
    I1Y  = get_mutual_info(loader, discriminators[0], 'modality_1',  n_classes)
    I2Y  = get_mutual_info(loader, discriminators[1], 'modality_2',  n_classes)
    I12Y = get_mutual_info(loader, discriminators[2], 'modality_12', n_classes)
    H1   = get_entropy(loader, entropy_estimators[0], 'modality_1')
    H2   = get_entropy(loader, entropy_estimators[1], 'modality_2')

    r_plus  = torch.minimum(H1, H2)
    r_minus = torch.minimum(H1 - I1Y, H2 - I2Y)
    r  = r_plus - r_minus
    u1 = I1Y  - r
    u2 = I2Y  - r
    s  = I12Y - r - u1 - u2

    r_adj, u1_adj, u2_adj, s_adj = RUS_adjustment([r, u1, u2, s])

    label = f"[{split_name}] " if split_name else ""
    print(f"{label}R={r_adj.mean():.4f}  "
          f"U_{args.mod0}={u1_adj.mean():.4f}  "
          f"U_{args.mod1}={u2_adj.mean():.4f}  "
          f"S={s_adj.mean():.4f}")
    return r, u1, u2, s


def normalize_pid(pid):
    pid_ = np.maximum(pid, 0)
    for i, pid_i in enumerate(pid_):
        sum_pid_i = np.sum(pid_i)
        if sum_pid_i > 1e-3:
            pid_[i] = pid_i / sum_pid_i
    return pid_


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Food101 CoMM LSMI-PID")
    print(f"  Image encoder : {args.img_encoder}  (img_size={args.img_size})")
    print(f"  Text  encoder : {args.txt_encoder}  (max_len={args.max_txt_len})")
    print(f"  mod0={args.mod0}  mod1={args.mod1}  embed_dim={args.embed_dim}")
    print(f"  Device        : {device}")
    print(f"{'='*60}\n")

    # ========= 1. TOKENIZER + DATA =========
    print("Loading tokenizer…")
    tokenizer = build_tokenizer(args.txt_encoder)

    print("Loading data…")
    traindata, valdata, testdata, train_clean, class_to_idx = get_dataloaders(
        args.data_root, tokenizer,
        bs=args.bs, num_workers=args.num_workers,
        img_size=args.img_size, max_txt_len=args.max_txt_len,
        val_split=args.val_split,
    )
    print(f"  train={len(traindata.dataset)}  val={len(valdata.dataset)}  "
          f"test={len(testdata.dataset)}  classes={len(class_to_idx)}")

    # ========= 2. BUILD + TRAIN CoMM =========
    print("\nBuilding model…")
    model = Food101CoMMModel(
        num_classes=NUM_CLASSES,
        img_encoder_type=args.img_encoder,
        txt_encoder_type=args.txt_encoder,
        embed_dim=args.embed_dim,
        freeze_encoders=args.freeze_encoders,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,}  (trainable: {train:,})")

    print("\nTraining…")
    model = train_comm(model, traindata, valdata,
                       epochs=args.epochs, lr=args.lr,
                       weight_decay=args.weight_decay,
                       save=args.saved_model)

    # ========= 3. TEST CoMM =========
    print("\nTest performance:")
    test_comm(model, testdata)

    # ========= 4. EXTRACT REPRESENTATIONS =========
    # Use clean (non-augmented) train loader so LSMI sees the true data distribution
    print("\nExtracting representations…")
    train_r1, train_r2, y_train = extract_representations(model, train_clean)
    test_r1,  test_r2,  y_test  = extract_representations(model, testdata)
    feat_dim_1 = train_r1.shape[1]
    feat_dim_2 = train_r2.shape[1]
    print(f"  Representation dim: {feat_dim_1}")

    # ========= 5. BUILD LSMI FEATURE LOADERS =========
    lsmi_train = DataLoader(
        feature_dataset(train_r1, train_r2, y_train),
        batch_size=args.lsmi_bs, shuffle=True,  num_workers=0)
    lsmi_test  = DataLoader(
        feature_dataset(test_r1,  test_r2,  y_test),
        batch_size=args.lsmi_bs, shuffle=False, num_workers=0)

    # ========= 6. TRAIN LSMI ESTIMATORS =========
    print("\nTraining LSMI discriminators…")
    discriminators = obtain_discriminator(
        lsmi_train, feat_dim_1, feat_dim_2,
        embed_size=args.lsmi_embed_size,
        n_classes=NUM_CLASSES,
        n_epochs=args.epochs_discriminator,
    )

    print("\nTraining LSMI entropy estimators…")
    entropy_estimators = obtain_entropy_estimator(
        lsmi_train, feat_dim_1, feat_dim_2,
        n_epochs=args.epochs_entropy,
    )

    # ========= 7. LSMI PID ESTIMATION =========
    print("\nPID:")
    LSMI_estimation(lsmi_train, discriminators, entropy_estimators, NUM_CLASSES, "train")
    r, u1, u2, s = LSMI_estimation(lsmi_test, discriminators, entropy_estimators, NUM_CLASSES, "test")

    # ========= 8. POINTWISE DISTRIBUTION =========
    # Stack as [U_mod0, U_mod1, R, S]
    pid = np.stack([u1.cpu().numpy(), u2.cpu().numpy(),
                    r.cpu().numpy(),  s.cpu().numpy()], axis=1)

    print(f"\nMean pointwise PID [U_{args.mod0}, U_{args.mod1}, R, S] (test):")
    print(np.mean(pid, axis=0))
    pid_norm = normalize_pid(pid)
    print("Normalised mean:", np.mean(pid_norm, axis=0))

    # ========= 9. CORRECTION (distribution level only) =========
    updated_pid = []
    for pid_i in pid:
        pid_i = list(pid_i)
        if pid_i[0] < 0 and pid_i[1] >= 0:
            pid_i[3] += pid_i[0];
            pid_i[0] = 0
        elif pid_i[1] < 0 and pid_i[0] >= 0:
            pid_i[3] += pid_i[1];
            pid_i[1] = 0
        updated_pid.append(pid_i)
    pid = np.array(updated_pid)

    print(f"\nAfter Correction Mean pointwise PID [U_{args.mod0}, U_{args.mod1}, R, S]:")
    print(np.mean(pid, axis=0))
    pid_norm = normalize_pid(pid)
    print("After Correction Normalised mean:", np.mean(pid_norm, axis=0))
