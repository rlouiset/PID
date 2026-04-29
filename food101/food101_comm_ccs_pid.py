"""
Food101 image-text multimodal classification with CoMM-inspired fusion.
Distribution-level CCS PID to measure R / U_img / U_txt / S.

Image encoders (--img-encoder):
  resnet50   — torchvision ResNet-50, ImageNet-1k V2 weights
               features before avgpool → (B, 2048, 7, 7) → 49 spatial tokens
  convnext   — torchvision ConvNeXt-Base, ImageNet-1k V1 weights
               (use timm with 'convnext_base.fb_in22k' for 22k weights)
               features → (B, 1024, 7, 7) → 49 spatial tokens
  blip_vit   — Salesforce/blip-image-captioning-base ViT-B/16 encoder
               last_hidden_state → (B, 197, 768) → projected tokens
               (trained at 384 px; use --img-size 384 for best quality)

Text encoders (--txt-encoder):
  bert       — bert-base-uncased     (768-d)
  roberta    — roberta-base          (768-d)
  deberta    — microsoft/deberta-v3-base (768-d)

Fusion: identical FusionTransformer to CoMM / trifeatures_comm_ccs_pid.py
  CLS token + concat(img_tokens, txt_tokens) → 1-layer pre-norm self-attention → CLS

PID: distribution-level CCS + Source redundancy → global R, U_img, U_txt, S
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

from utils_ours import return_redundancy_test_performances

multiprocessing.set_start_method('fork', force=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

softmax = nn.Softmax(dim=-1)


# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data-root",    default="/home/rlouiset/Food101", type=str)
parser.add_argument("--img-size",     default=224, type=int,
                    help="Resize images to this size (use 384 with blip_vit for best quality)")
parser.add_argument("--max-txt-len",  default=64,  type=int,
                    help="Max tokenization length for text titles")
parser.add_argument("--val-split",    default=0.1, type=float,
                    help="Fraction of training data used as validation set")
# Encoders
parser.add_argument("--img-encoder",  default="convnext",
                    choices=["resnet50", "convnext", "blip_vit"])
parser.add_argument("--txt-encoder",  default="roberta",
                    choices=["bert", "roberta", "deberta"])
parser.add_argument("--freeze-encoders", action="store_true",
                    help="Freeze pretrained encoders, train only adapters + fusion + heads")
# Model
parser.add_argument("--embed-dim",    default=512, type=int)
parser.add_argument("--bs",           default=32,  type=int)
parser.add_argument("--num-workers",  default=4,   type=int)
parser.add_argument("--epochs",       default=20,  type=int)
parser.add_argument("--lr",           default=1e-4, type=float)
parser.add_argument("--weight-decay", default=1e-2, type=float)
parser.add_argument("--saved-model",  default=None, type=str)
args = parser.parse_args()

NUM_CLASSES = 101


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class Food101Dataset(Dataset):
    """
    CSV format (no header): filename, web_title, class_label
    Images:  data_root/images/{split}/{class_label}/{filename}
    """

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
        img_path = os.path.join(self.img_root, cls, filename)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        enc = self.tokenizer(
            title,
            max_length=self.max_txt_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids   = enc['input_ids'].squeeze(0)
        attn_mask   = enc['attention_mask'].squeeze(0)
        label       = self.class_to_idx[cls]
        return img, input_ids, attn_mask, label


def food101_collate(batch):
    imgs      = torch.stack([b[0] for b in batch])
    input_ids = torch.stack([b[1] for b in batch])
    attn_mask = torch.stack([b[2] for b in batch])
    labels    = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return imgs, input_ids, attn_mask, labels


def get_dataloaders(data_root, tokenizer, bs, num_workers, img_size=224,
                    max_txt_len=64, val_split=0.1):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
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

    train_ds = Subset(ds_aug,   tr_idx)
    val_ds   = Subset(ds_noaug, va_idx)   # no augmentation for val
    test_ds  = Food101Dataset(txt_test_csv, img_test_root, tokenizer, test_tf, max_txt_len,
                              class_to_idx=class_to_idx)

    loader_kw = dict(batch_size=bs, num_workers=num_workers,
                     collate_fn=food101_collate, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kw),
        DataLoader(val_ds,   shuffle=False, **loader_kw),
        DataLoader(test_ds,  shuffle=False, **loader_kw),
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
    """
    Returns (encoder_module, out_channels_or_dim, is_spatial_map).
    is_spatial_map=True  → output is (B, C, H, W), use SpatialTokenAdapter
    is_spatial_map=False → output is (B, T, D), use SequenceTokenAdapter
    """
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
    """Returns (encoder_module, hidden_dim)."""
    hub = {
        "bert":    "bert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base",
    }
    from transformers import AutoModel
    m = AutoModel.from_pretrained(hub[name])
    return HFTextEncoder(m), 768


def build_tokenizer(txt_encoder_name):
    hub = {
        "bert":    "bert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base",
    }
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hub[txt_encoder_name])


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
    """(B, C, H, W) → (B, H*W, embed_dim) with 2D sincos pos embedding."""
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
        return (tokens + pos).flatten(2).transpose(1, 2)   # (B, H*W, embed_dim)


class SequenceTokenAdapter(nn.Module):
    """(B, T, in_dim) → (B, T, embed_dim)."""
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class FusionTransformer(nn.Module):
    """
    CLS token + concat(img_tokens, txt_tokens) → 1-layer pre-norm self-attention → CLS.
    Identical to the FusionTransformer in trifeatures_comm_ccs_pid.py.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class Food101CoMMModel(nn.Module):
    def __init__(self, num_classes=101, img_encoder_type="convnext",
                 txt_encoder_type="roberta", embed_dim=512,
                 freeze_encoders=False):
        super().__init__()

        img_enc, img_dim, img_spatial = build_img_encoder(img_encoder_type)
        txt_enc, txt_dim              = build_txt_encoder(txt_encoder_type)

        self.img_encoder = img_enc
        self.txt_encoder = txt_enc
        self.img_spatial = img_spatial

        if img_spatial:
            self.img_adapter = SpatialTokenAdapter(img_dim, embed_dim, image_size=7)
        else:
            self.img_adapter = SequenceTokenAdapter(img_dim, embed_dim)
        self.txt_adapter = SequenceTokenAdapter(txt_dim, embed_dim)

        self.fusion    = FusionTransformer(width=embed_dim, n_heads=8, n_layers=1)
        self.head      = nn.Linear(embed_dim, num_classes)
        self.mod_heads = nn.ModuleList([nn.Linear(embed_dim, num_classes),
                                        nn.Linear(embed_dim, num_classes)])
        self.reps = []   # mean-pooled (B, embed_dim) per modality, set in forward

        if freeze_encoders:
            for p in self.img_encoder.parameters(): p.requires_grad = False
            for p in self.txt_encoder.parameters(): p.requires_grad = False

    def forward(self, imgs, input_ids, attention_mask):
        img_feat   = self.img_encoder(imgs)
        if self.img_spatial:
            img_tokens = self.img_adapter(img_feat)          # (B, 49, D)
        else:
            img_tokens = self.img_adapter(img_feat)          # (B, T, D)

        txt_feat   = self.txt_encoder(input_ids, attention_mask)
        txt_tokens = self.txt_adapter(txt_feat)              # (B, seq_len, D)

        self.reps  = [img_tokens.mean(dim=1),                # (B, D) each
                      txt_tokens.mean(dim=1)]

        fused        = self.fusion([img_tokens, txt_tokens]) # (B, D)
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
        total_loss = 0.; n = 0
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
        val_loss = 0.; vn = 0; vc = 0
        with torch.no_grad():
            for imgs, inp, msk, y in validdata:
                imgs, inp, msk, y = (imgs.to(device), inp.to(device),
                                     msk.to(device), y.to(device).long())
                joint, _, _ = model(imgs, inp, msk)
                val_loss += criterion(joint, y).item() * len(y)
                vn += len(y)
                vc += (joint.argmax(1) == y).sum().item()
        val_loss /= vn
        print(f"Epoch {epoch:02d}  train={total_loss/n:.4f}  "
              f"val={val_loss:.4f}  acc={vc/vn:.4f}")
        if val_loss < best_val:
            best_val   = val_loss
            best_model = copy.deepcopy(model)
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
                                 msk.to(device), y.to(device).long())
            joint, _, mod_logits = model(imgs, inp, msk)
            tj  += criterion(joint,         y).item() * len(y)
            tm0 += criterion(mod_logits[0], y).item() * len(y)
            tm1 += criterion(mod_logits[1], y).item() * len(y)
            n   += len(y)
            pj.append(joint.cpu());      pm0.append(mod_logits[0].cpu())
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
    """Extract mean-pooled (N, embed_dim) representations for each modality."""
    model.eval()
    r0, r1, tgts = [], [], []
    with torch.no_grad():
        for imgs, inp, msk, y in loader:
            _ = model(imgs.to(device), inp.to(device), msk.to(device))
            r0.append(model.reps[0].cpu())
            r1.append(model.reps[1].cpu())
            tgts.append(y.cpu())
    X = {"modality0": torch.cat(r0).float(), "modality1": torch.cat(r1).float()}
    return X, torch.cat(tgts).long().float()


# ═══════════════════════════════════════════════════════════════════════════════
# PID UTILITIES  (same as humor_ccs_source_redundancy.py)
# ═══════════════════════════════════════════════════════════════════════════════

def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):
    probs = torch.clamp(probs, min=eps, max=1.0)
    ce  = -torch.log(probs)[torch.arange(len(targets)), targets.long()].mean()
    acc = (probs.argmax(dim=1) == targets).float().mean()
    return acc.item(), ce.item()


def compute_log_py(targets, num_classes):
    counts = torch.bincount(targets.long(), minlength=num_classes).float()
    probs  = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return torch.log(probs)[targets.long()]


def ce_per_sample(targets, probs, eps=1e-12):
    probs = torch.clamp(probs, eps, 1.0)
    return -torch.log(probs[torch.arange(len(targets)), targets.long()])


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


def compute_pid_global(joint_ce, mod0_ce, mod1_ce, red_ce, src_red_ce, targets,
                       num_classes, mod0_name="image", mod1_name="text"):
    hy = compute_entropy_from_targets(targets, num_classes)
    print(f"H(Y)={hy:.4f}  joint={joint_ce:.4f}  red={red_ce:.4f}  "
          f"src_red={src_red_ce:.4f}  {mod0_name}={mod0_ce:.4f}  "
          f"{mod1_name}={mod1_ce:.4f}")

    mod0_ce    = min(mod0_ce,    hy);  mod1_ce    = min(mod1_ce,    hy)
    red_ce     = min(red_ce,     hy);  src_red_ce = min(src_red_ce, hy)
    red_ce     = max(red_ce,     joint_ce, mod0_ce, mod1_ce)
    src_red_ce = max(src_red_ce, joint_ce, mod0_ce, mod1_ce)
    red_ce     = min(red_ce,     src_red_ce)
    mod0_ce    = min(max(mod0_ce, joint_ce), red_ce)
    mod1_ce    = min(max(mod1_ce, joint_ce), red_ce)

    i_total = hy - joint_ce
    i_r     = hy - red_ce;    i_r_src = hy - src_red_ce
    i_u0    = (hy - mod0_ce) - i_r
    i_u1    = (hy - mod1_ce) - i_r
    i_s     = i_total - i_u0 - i_u1 - i_r

    if i_s < 0:
        i_r     -= i_s;  i_r_src -= i_s
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
        print(f"{name:<14s} | {results[key]['accuracy']:8.4f} | {results[key]['cross_entropy']:8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Food101 CoMM CCS-PID")
    print(f"  Image encoder : {args.img_encoder}  (img_size={args.img_size})")
    print(f"  Text  encoder : {args.txt_encoder}  (max_len={args.max_txt_len})")
    print(f"  Embed dim     : {args.embed_dim}")
    print(f"  Freeze enc.   : {args.freeze_encoders}")
    print(f"  Device        : {device}")
    print(f"{'='*60}\n")

    # ========= 1. TOKENIZER =========
    print("Loading tokenizer…")
    tokenizer = build_tokenizer(args.txt_encoder)

    # ========= 2. DATA =========
    print("Loading data…")
    traindata, valdata, testdata, class_to_idx = get_dataloaders(
        args.data_root, tokenizer,
        bs=args.bs, num_workers=args.num_workers,
        img_size=args.img_size, max_txt_len=args.max_txt_len,
        val_split=args.val_split,
    )
    print(f"  train={len(traindata.dataset)}  val={len(valdata.dataset)}  "
          f"test={len(testdata.dataset)}  classes={len(class_to_idx)}")

    # ========= 3. MODEL =========
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

    # ========= 4. TRAIN =========
    print("\nTraining…")
    model = train_comm(model, traindata, valdata,
                       epochs=args.epochs, lr=args.lr,
                       weight_decay=args.weight_decay,
                       save=args.saved_model)

    # ========= 5. TEST =========
    print("\nTest performance:")
    d = test_comm(model, testdata)
    print_model_metrics(d)

    # ========= 6. EXTRACT REPRESENTATIONS =========
    print("\nExtracting representations…")
    X_train, y_train = extract_split(model, traindata)
    X_val,   y_val   = extract_split(model, valdata)
    X_test,  y_test  = extract_split(model, testdata)
    print(f"  embed_dim={X_train['modality0'].shape[1]}  "
          f"train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    # ========= 7. CCS REDUNDANCY =========
    targets = d["true_labels"].long()
    log_py  = compute_log_py(targets, NUM_CLASSES)
    _, ce_list = compute_probs_and_ce(d["pred_modalities"], targets)

    i_list    = [-ce - log_py for ce in ce_list]
    same_sign = torch.sign(i_list[0]) == torch.sign(i_list[1])
    worst_ce  = torch.max(torch.stack(ce_list, dim=1), dim=1).values
    ccs       = torch.where(same_sign, worst_ce, -log_py)

    d["redundancy_ce"]           = ccs.mean().item()
    d["redundancy_pointwise_ce"] = ccs.numpy()

    # ========= 8. SOURCE REDUNDANCY =========
    print("\nFitting source redundancy model…")
    y_pred_dict = return_redundancy_test_performances(
        X_train, X_val, X_test,
        y_train.long(), y_val.long(), y_test.long(),
        f"food101_{args.img_encoder}_{args.txt_encoder}",
        distribution_target="categorical",
        num_classes=NUM_CLASSES,
    )

    results = compute_redundancy_metrics(y_pred_dict)
    print_redundancy_metrics(results)

    d["source_redundancy_pointwise_ce"] = results["average"]["cross_entropy"]
    d["source_redundancy_preds"]        = y_pred_dict["average"]

    # ========= 9. GLOBAL PID =========
    print("\nGlobal PID:")
    compute_pid_global(
        d["joint_ce"],
        d["modalities_ce"][0],
        d["modalities_ce"][1],
        d["redundancy_ce"],
        d["source_redundancy_pointwise_ce"],
        targets=d["true_labels"],
        num_classes=NUM_CLASSES,
        mod0_name="image",
        mod1_name="text",
    )
