"""
CoMM architecture + CCS/Source PID estimator on the TriFeatures bimodal dataset.

Two experimental cases (run via --task / --biased):
  Case 1 — Redundancy   : --task share  --num-classes 10          (biased=False)
            Both images share the same shape → high R, near-zero S
  Case 2 — Synergy      : --task synergy --biased --num-classes 2
            Texture(mod1)×Color(mod2) correlation detectable only jointly → high S

Architecture: exact same CoMMEncoder (Conv1d → 5-layer Transformer, pre-norm)
              + FusionTransformer (CLS token, 1-layer self-attention)
              as used in the affect scripts, prepended by a lightweight CNN backbone
              that converts each 224×224 image into a sequence of spatial tokens.

Dataset: BimodalTrifeatures — images are generated once and cached in --data-root.
         Source: Duplums/CoMM  https://github.com/Duplums/CoMM/blob/main/dataset/trifeatures.py
"""

from __future__ import print_function
import argparse
import copy
import math
import multiprocessing
import os
import re

import numpy as np
import scipy.ndimage
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


# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data-root",    default="/home/rlouiset/trifeatures_data", type=str,
                    help="Directory to store / load generated PNG images")
parser.add_argument("--task",         default="share",
                    choices=["share", "unique1", "unique2", "synergy"])
parser.add_argument("--biased",       action="store_true",
                    help="Use biased (synergistic) dataset pairing")
parser.add_argument("--num-classes",  default=10,  type=int,
                    help="10 for share/unique tasks, 2 for synergy")
parser.add_argument("--bs",           default=64,  type=int)
parser.add_argument("--num-workers",  default=4,   type=int)
parser.add_argument("--embed-dim",    default=40,  type=int,
                    help="Transformer embedding dim (same as CoMM affect scripts)")
parser.add_argument("--cnn-channels", default=128, type=int,
                    help="CNN backbone output channels (= CoMMEncoder input dim)")
parser.add_argument("--epochs",       default=50,  type=int)
parser.add_argument("--lr",           default=1e-3, type=float)
parser.add_argument("--weight-decay", default=1e-2, type=float)
parser.add_argument("--saved-model",  default=None, type=str)
args = parser.parse_args()

NUM_CLASSES = args.num_classes


# ═══════════════════════════════════════════════════════════════════════════════
# TRIFEATURES DATASET
# Adapted from https://github.com/Duplums/CoMM/blob/main/dataset/trifeatures.py
# ═══════════════════════════════════════════════════════════════════════════════

class Trifeatures(Dataset):
    """
    1 000 images (10 shapes × 10 colors × 10 textures), each rendered
    num_per_combination times. 80 % train / 20 % test split.
    """
    BASE_SHAPES   = ["triangle", "square", "plus", "circle", "tee",
                     "rhombus", "pentagon", "star", "fivesquare", "trapezoid"]
    BASE_COLORS   = {
        "red": (1., 0., 0.), "green": (0., 1., 0.), "blue": (0., 0., 1.),
        "yellow": (1., 1., 0.), "pink": (1., 0.4, 1.), "cyan": (0., 1., 1.),
        "purple": (0.3, 0., 0.5), "ocean": (0.1, 0.4, 0.5),
        "orange": (1., 0.6, 0.), "white": (1., 1., 1.),
    }
    BASE_TEXTURES = ["solid", "stripes", "grid", "hexgrid", "dots", "noise",
                     "triangles", "zigzags", "rain", "pluses"]
    BG_COLOR      = np.array((0.5, 0.5, 0.5), dtype=np.float32)
    BASE_COLORS   = {n: np.array(c, dtype=np.float32) for n, c in BASE_COLORS.items()}
    BASE_SIZE     = 128
    RENDER_SIZE   = 224
    RANDOM_ANGLE_RANGE = 45
    TEXTURE_SCALE = 10

    def __init__(self, root, split="train", num_per_combination=3,
                 transform=None, seed=42):
        self.root               = root
        self.split              = split
        self.num_per_combination = num_per_combination
        self.transform          = transform
        self.split_ratio        = 0.8
        self.rng                = np.random.default_rng(seed)
        self._base_templates    = {s: self._render_plain_shape(s)
                                   for s in self.BASE_SHAPES}

        if not self._check_integrity():
            self.generate_data()

        self.root         = os.path.join(self.root, self.split)
        self.name_images  = sorted(f for f in os.listdir(self.root) if f.endswith('.png'))
        self.color_to_idx   = {n: i for i, n in enumerate(self.BASE_COLORS)}
        self.shape_to_idx   = {n: i for i, n in enumerate(self.BASE_SHAPES)}
        self.texture_to_idx = {n: i for i, n in enumerate(self.BASE_TEXTURES)}

    def __len__(self):
        return len(self.name_images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.name_images[idx]))
        if self.transform is not None:
            img = self.transform(img)
        return img, self._target(self.name_images[idx])

    def _target(self, filename):
        parts = re.split(r'_', filename.split('.')[0])   # shape_texture_color[_id].png
        return (self.shape_to_idx[parts[0]],
                self.color_to_idx[parts[2]],
                self.texture_to_idx[parts[1]])

    def _check_integrity(self):
        pattern = re.compile(r"(\w+)_(\w+)_(\w+)(_[0-9]+)?\.png")
        total = len(self.BASE_SHAPES) * len(self.BASE_COLORS) * len(self.BASE_TEXTURES)
        expected = {"train": int(self.split_ratio * self.num_per_combination * total),
                    "test":  total - int(self.split_ratio * total)}
        for split, exp in expected.items():
            path = os.path.join(self.root, split)
            if not os.path.exists(path):
                return False
            n = sum(1 for f in os.listdir(path) if pattern.match(f))
            if n != exp:
                return False
            print(f"{n} images found in {split}.")
        return True

    def generate_data(self):
        print("Generating TriFeatures images (this only happens once)…")
        split_arr = self._split_array()
        os.makedirs(os.path.join(self.root, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "test"),  exist_ok=True)
        idx = 0
        for s in self.BASE_SHAPES:
            for t in self.BASE_TEXTURES:
                for c in self.BASE_COLORS:
                    img_arr = self._render_stimulus(s, c, t)
                    img = Image.fromarray((img_arr * 255.).astype(np.uint8), mode='RGB')
                    if split_arr[idx] == 1:
                        img.save(os.path.join(self.root, "test", f"{s}_{t}_{c}.png"))
                    else:
                        for i in range(self.num_per_combination):
                            img_arr = self._render_stimulus(s, c, t)
                            img = Image.fromarray((img_arr * 255.).astype(np.uint8), 'RGB')
                            img.save(os.path.join(self.root, "train", f"{s}_{t}_{c}_{i}.png"))
                    idx += 1
        print("Dataset generated.")

    def _split_array(self):
        total = len(self.BASE_SHAPES) * len(self.BASE_COLORS) * len(self.BASE_TEXTURES)
        arr = np.array([1] * (total - int(total * self.split_ratio))
                       + [0] * int(total * self.split_ratio))
        self.rng.shuffle(arr)
        return arr

    # ---------- rendering ----------

    @staticmethod
    def _render_plain_shape(name):
        size = Trifeatures.BASE_SIZE
        shape = np.zeros([size, size], np.float32)
        if name == "square":
            shape[:, :] = 1.
        elif name == "circle":
            for i in range(size):
                for j in range(size):
                    if (i+.5-size//2)**2 + (j+.5-size//2)**2 < (size//2)**2:
                        shape[i, j] = 1.
        elif name == "triangle":
            for i in range(size):
                for j in range(size):
                    if abs(j - size//2) - abs(i//2) < 1:
                        shape[i, j] = 1.
        elif name == "plus":
            shape[:, size//2-size//6:size//2+size//6+1] = 1.
            shape[size//2-size//6:size//2+size//6+1, :] = 1.
        elif name == "tee":
            shape[:, size//2-size//6:size//2+size//6+1] = 1.
            shape[:size//3, :] = 1.
        elif name == "rhombus":
            for i in range(size):
                for j in range(size):
                    if 0 < j - size//2 + i//2 < size//2:
                        shape[i, j] = 1.
        elif name == "pentagon":
            ml = int(size * 0.4)
            for i in range(ml):
                for j in range(size):
                    if abs(j - size//2) - abs(i * 1.25) < 1:
                        shape[i, j] = 1.
            for i in range(ml, size):
                xo = (i - ml) / 3.1
                for j in range(size):
                    if xo < j < size - xo:
                        shape[i, j] = 1.
        elif name == "star":
            line  = int(size * 0.4)
            line2 = line + int(0.2 * size)
            line3 = line + int(0.15 * size)
            for i in range(line):
                for j in range(size):
                    if abs(j - size//2) - abs(i//4) < 1:
                        shape[i, j] = 1.
            for i in range(line, line2):
                xo = (i - line) * 2.4
                for j in range(size):
                    if xo < j < size - xo:
                        shape[i, j] = 1.
            for i in range(line3, size):
                xo1 = size*0.33 - 0.43*(i-line3)
                xo2 = size*0.62 - 1.05*(i-line3)
                for j in range(size):
                    if xo1 < j < xo2 or xo1 < size-j < xo2:
                        shape[i, j] = 1.
        elif name == "fivesquare":
            shape[:, :] = 1.
            shape[:, size//3:2*size//3] = 0.
            shape[size//3:2*size//3, :] = 0.
            shape[size//3:2*size//3, size//3:2*size//3] = 1.
        elif name == "trapezoid":
            for i in range(size):
                xo = i / 3.1
                for j in range(size):
                    if xo < j < size - xo:
                        shape[i, j] = 1.
        return shape

    def get_texture(self, size, name):
        sc  = self.TEXTURE_SCALE
        lw  = sc // 3
        slw = sc // 5
        ox  = int(self.rng.integers(0, sc))
        oy  = int(self.rng.integers(0, sc))
        tex = np.zeros([size, size], dtype=np.float32)
        if name == "solid":
            return np.ones_like(tex)
        elif name == "stripes":
            for i in range(size):
                if (i + oy) % sc < lw:
                    tex[i, :] = 1.
        elif name == "grid":
            for i in range(size):
                if (i + oy) % sc < lw:
                    tex[i, :] = 1.
            for j in range(size):
                if (j + ox) % sc < lw:
                    tex[:, j] = 1.
        elif name == "hexgrid":
            for i in range(size):
                for j in range(size):
                    y = i + oy; x = j + ox
                    if ((x + int(1.73*y)) % sc < slw
                            or (x - int(1.73*y)) % sc < slw
                            or y % sc < slw):
                        tex[i, j] = 1.
        elif name == "dots":
            r2 = (3*sc//7)**2
            for i in range(size):
                for j in range(size):
                    y = (i+oy)%sc - sc//2; x = (j+ox)%sc - sc//2
                    if x*x + y*y < r2:
                        tex[i, j] = 1.
        elif name == "noise":
            tex = self.rng.binomial(1, 0.5, tex.shape).astype(np.float32)
        elif name == "triangles":
            for i in range(size):
                for j in range(size):
                    y = (i+oy)%sc; x = (j+ox)%sc
                    if y//2 - abs(x - sc//2) > 0:
                        tex[i, j] = 1.
        elif name == "zigzags":
            so = sc - sc//2
            for i in range(size):
                ss = ((i+oy)//sc) % 2
                for j in range(size):
                    y = (i+oy)%sc; x = (j+ox)%sc
                    if ss: x = sc - x - 1
                    off = y // 2
                    if off < x < so + off:
                        tex[i, j] = 1.
        elif name == "rain":
            rh = sc - sc//3
            for i in range(size):
                for j in range(size):
                    if self.rng.binomial(1, 0.05):
                        tex[i:i+rh, j:j+1] = 1.
        elif name == "pluses":
            hw = 1.5
            for i in range(size):
                ss = ((i+oy)//sc) % 2
                for j in range(size):
                    y = (i+oy)%sc; x = (j+ox)%sc
                    if ss:
                        if (abs(x) < hw or sc-x < hw
                                or (abs(y-sc//2) < hw and abs(x-sc//2) > hw)):
                            tex[i, j] = 1.
                    else:
                        if abs(x-sc//2) < hw or abs(y-sc//2) < hw:
                            tex[i, j] = 1.
        return tex

    def _render_uncolored_shape(self, name):
        tmpl  = self._base_templates[name]
        angle = int(self.rng.integers(-self.RANDOM_ANGLE_RANGE, self.RANDOM_ANGLE_RANGE))
        shape = scipy.ndimage.rotate(tmpl, angle, order=1)
        ns    = shape.shape
        img   = np.zeros([self.RENDER_SIZE, self.RENDER_SIZE], np.float32)
        ox    = int(self.rng.integers(0, self.RENDER_SIZE - ns[0]))
        oy    = int(self.rng.integers(0, self.RENDER_SIZE - ns[1]))
        img[ox:ox+ns[0], oy:oy+ns[1]] = shape
        return img

    def _render_stimulus(self, shape, color, texture):
        img   = self._render_uncolored_shape(shape)
        ts    = 2 * self.RENDER_SIZE
        tex   = self.get_texture(ts, texture)
        angle = int(self.rng.integers(-self.RANDOM_ANGLE_RANGE, self.RANDOM_ANGLE_RANGE))
        tex   = scipy.ndimage.rotate(tex, angle, order=0, reshape=False)
        tex   = tex[self.RENDER_SIZE//2:-self.RENDER_SIZE//2,
                    self.RENDER_SIZE//2:-self.RENDER_SIZE//2]
        img   = np.multiply(img, tex)
        ci    = img[:, :, None] * self.BASE_COLORS[color][None, None, :]
        ci   += (1 - img)[:, :, None] * self.BG_COLOR[None, None, :]
        return ci


class BimodalTrifeatures(Trifeatures):
    """
    Pairs of images sharing one attribute (share_attr).
    Tasks:
      "share"   → shared attribute label (10 classes by default)
      "unique1" → unique attr of image 1  (10 classes)
      "unique2" → unique attr of image 2  (10 classes)
      "synergy" → 1 iff correlated texture-color pair co-occurs (2 classes)
    biased=True  → pairs must also satisfy a fixed texture-color correlation (synergy dataset)
    biased=False → any pair sharing the shape (redundancy+uniqueness dataset)
    """
    def __init__(self, root, split="train", task="share",
                 share_attr="shape", unique_attr="texture",
                 synergy_attr=("texture", "color"),
                 biased=False, max_size=10000,
                 num_per_combination=3, transform=None, seed=42):
        super().__init__(root, split, num_per_combination=num_per_combination,
                         transform=transform, seed=seed)
        self.task        = task
        self.share_attr  = share_attr
        self.unique_attr = unique_attr
        self.synergy_attr = synergy_attr
        self.biased      = biased
        self.max_size    = int(max_size)

        attrs = dict(color=self.color_to_idx, shape=self.shape_to_idx,
                     texture=self.texture_to_idx)
        synergy_vals = (sorted(attrs[synergy_attr[0]].values()),
                        sorted(attrs[synergy_attr[1]].values()))
        perm = self.rng.permutation(synergy_vals[1])
        self.correlated_feature_pairs = list(zip(synergy_vals[0], perm))
        self.idx_pairs = self._get_idx_pairs()

    def _get_idx_pairs(self):
        a2i = dict(shape=0, color=1, texture=2)
        n   = len(self.name_images)
        targets = np.array([self._target(nm) for nm in self.name_images])

        share_vals = targets[:, a2i[self.share_attr]]
        share_eq   = (share_vals.reshape(n, 1) == share_vals.reshape(1, n))

        # Always compute synergy mask (needed for balanced synergy task)
        sv0 = targets[:, a2i[self.synergy_attr[0]]]
        sv1 = targets[:, a2i[self.synergy_attr[1]]]
        synergy_eq = np.zeros((n, n), dtype=bool)
        for p in self.correlated_feature_pairs:
            m0 = (sv0.reshape(n, 1) == p[0]).repeat(n, axis=1)
            m1 = (sv1.reshape(1, n) == p[1]).repeat(n, axis=0)
            synergy_eq |= (m0 & m1)

        if self.task == "synergy":
            # Supervised synergy classification: balance positive (label=1) and
            # negative (label=0) pairs so that H(Y) > 0 and the model can learn.
            # biased flag is ignored here — both classes must be present.
            pos = np.argwhere(share_eq & synergy_eq)
            neg = np.argwhere(share_eq & ~synergy_eq)
            n_each = min(len(pos), len(neg), self.max_size // 2)
            pos_sel = self.rng.choice(len(pos), size=n_each, replace=False)
            neg_sel = self.rng.choice(len(neg), size=n_each, replace=False)
            pairs = np.vstack([pos[pos_sel], neg[neg_sel]])
            self.rng.shuffle(pairs)
            return pairs

        allowed = np.argwhere(share_eq & (synergy_eq if self.biased
                                          else np.ones((n, n), dtype=bool)))
        n_allowed = len(allowed)
        if self.max_size > n_allowed:
            self.max_size = n_allowed
        sel = self.rng.choice(n_allowed, size=self.max_size, replace=False)
        return allowed[sel]

    def __len__(self):
        return len(self.idx_pairs)

    def __getitem__(self, idx):
        i1, i2 = self.idx_pairs[idx]
        im1, t1 = super().__getitem__(i1)
        im2, t2 = super().__getitem__(i2)
        a2i = dict(shape=0, color=1, texture=2)
        if self.task == "share":
            return [im1, im2], int(t1[a2i[self.share_attr]])
        elif self.task == "unique1":
            return [im1, im2], int(t1[a2i[self.unique_attr]])
        elif self.task == "unique2":
            return [im1, im2], int(t2[a2i[self.unique_attr]])
        elif self.task == "synergy":
            key = (int(t1[a2i[self.synergy_attr[0]]]),
                   int(t2[a2i[self.synergy_attr[1]]]))
            return [im1, im2], int(key in self.correlated_feature_pairs)
        raise ValueError(self.task)


def bimodal_collate(batch):
    """Collate ([img1, img2], label) → (tensor_img1, tensor_img2, tensor_labels)."""
    imgs0  = torch.stack([b[0][0] for b in batch])
    imgs1  = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1]   for b in batch], dtype=torch.long)
    return imgs0, imgs1, labels


def get_dataloaders(data_root, task, biased, bs, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    kw = dict(task=task, biased=biased, num_per_combination=3)
    train_ds = BimodalTrifeatures(data_root, split="train", transform=train_tf, **kw)
    test_ds  = BimodalTrifeatures(data_root, split="test",  transform=test_tf,  **kw)

    loader_kw = dict(batch_size=bs, collate_fn=bimodal_collate,
                     num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
# CoMM ARCHITECTURE (exact same as affect scripts)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageBackbone(nn.Module):
    """
    Lightweight CNN: (B, 3, 224, 224) → (B, T, cnn_channels) spatial token sequence.
    Uses AdaptiveAvgPool2d(4) → T = 16 spatial tokens, matching CoMMEncoder input.
    """
    def __init__(self, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,           32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2),                                                            # 112
            nn.Conv2d(32,          64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),                                                            # 56
            nn.Conv2d(64,          128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                                            # 28
            nn.Conv2d(128, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                                    # 4×4
        )

    def forward(self, x):
        f = self.net(x)                              # (B, C, 4, 4)
        B, C, H, W = f.shape
        return f.view(B, H * W, C)                   # (B, 16, C)


def _build_sincos_posemb(max_seq_len, embed_dim):
    pe  = torch.zeros(max_seq_len, embed_dim)
    pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class CoMMEncoder(nn.Module):
    """Conv1d projection → 5-layer pre-norm Transformer. Input: (B, T, n_features)."""
    def __init__(self, n_features, embed_dim=40, max_seq_len=16,
                 n_heads=5, n_layers=5, positional_encoding=False):
        super().__init__()
        self.conv      = nn.Conv1d(n_features, embed_dim, kernel_size=1, bias=False)
        self.use_pe    = positional_encoding
        if positional_encoding:
            self.register_buffer("pos_emb", _build_sincos_posemb(max_seq_len, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):                            # x: (B, T, n_features)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)   # → (B, T, embed_dim)
        if self.use_pe:
            x = x + self.pos_emb[:, :x.size(1)]
        return self.transformer(x)


class FusionTransformer(nn.Module):
    """CLS + concat modality sequences → 1-layer self-attention → CLS output."""
    def __init__(self, embed_dim=40, n_heads=8, n_layers=1):
        super().__init__()
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, sequences):
        B   = sequences[0].size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls] + sequences, dim=1)
        return self.norm(self.transformer(tok)[:, 0])


class TriFeaturesCoMMModel(nn.Module):
    """
    Per-modality: ImageBackbone → CoMMEncoder
    Fusion:       FusionTransformer (CLS) → joint classifier
    Per-modality heads: mean-pool sequence → unimodal classifier
    """
    def __init__(self, num_classes, embed_dim=40, cnn_channels=128,
                 seq_len=16, positional_encoding=False):
        super().__init__()
        self.backbones = nn.ModuleList([
            ImageBackbone(cnn_channels), ImageBackbone(cnn_channels)
        ])
        self.encoders = nn.ModuleList([
            CoMMEncoder(cnn_channels, embed_dim, seq_len,
                        positional_encoding=positional_encoding),
            CoMMEncoder(cnn_channels, embed_dim, seq_len,
                        positional_encoding=positional_encoding),
        ])
        self.fusion    = FusionTransformer(embed_dim)
        self.head      = nn.Linear(embed_dim, num_classes)
        self.mod_heads = nn.ModuleList([nn.Linear(embed_dim, num_classes),
                                        nn.Linear(embed_dim, num_classes)])
        self.reps    = []
        self.fuseout = None

    def forward(self, inputs):
        tokens = [bb(x.float()) for bb, x in zip(self.backbones, inputs)]
        seqs   = [enc(t) for enc, t in zip(self.encoders, tokens)]
        self.reps    = seqs
        fused        = self.fusion(seqs)
        self.fuseout = fused
        joint_logits = self.head(fused)
        mod_logits   = [h(s.mean(dim=1)) for h, s in zip(self.mod_heads, seqs)]
        return joint_logits, seqs, mod_logits


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def train_comm(model, traindata, validdata, epochs, lr, weight_decay, save=None):
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_model = copy.deepcopy(model)
    best_val   = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.; n = 0
        for m0, m1, y in traindata:
            inputs = [m0.to(device), m1.to(device)]
            y = y.to(device).long()
            joint, _, mod_logits = model(inputs)
            loss = criterion(joint, y) + sum(criterion(l, y) for l in mod_logits)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()
            total_loss += loss.item() * len(y); n += len(y)

        model.eval()
        val_loss = 0.; vn = 0; vc = 0
        with torch.no_grad():
            for m0, m1, y in validdata:
                inputs = [m0.to(device), m1.to(device)]
                y = y.to(device).long()
                joint, _, _ = model(inputs)
                val_loss += criterion(joint, y).item() * len(y)
                vn       += len(y)
                vc       += (joint.argmax(1) == y).sum().item()
        val_loss /= vn
        print(f"Epoch {epoch}  train={total_loss/n:.4f}  val={val_loss:.4f}  acc={vc/vn:.4f}")
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
        for m0, m1, y in testdata:
            inputs = [m0.to(device), m1.to(device)]
            y = y.to(device).long()
            joint, _, mod_logits = model(inputs)
            tj  += criterion(joint,         y).item() * len(y)
            tm0 += criterion(mod_logits[0], y).item() * len(y)
            tm1 += criterion(mod_logits[1], y).item() * len(y)
            n   += len(y)
            pj.append(joint.cpu()); pm0.append(mod_logits[0].cpu())
            pm1.append(mod_logits[1].cpu()); ys.append(y.cpu())

    pj  = torch.cat(pj);  pm0 = torch.cat(pm0)
    pm1 = torch.cat(pm1); ys  = torch.cat(ys)
    return {
        "joint_acc":       (pj.argmax(1) == ys).float().mean().item(),
        "modalities_acc": [(pm0.argmax(1) == ys).float().mean().item(),
                           (pm1.argmax(1) == ys).float().mean().item()],
        "joint_ce":       tj / n,
        "modalities_ce":  [tm0 / n, tm1 / n],
        "pred_joint":      pj,
        "pred_modalities": [pm0, pm1],
        "true_labels":     ys,
    }


def extract_split(model, loader):
    """Mean-pool CoMM sequences → (N, embed_dim) per modality."""
    model.eval()
    r0, r1, tgts = [], [], []
    with torch.no_grad():
        for m0, m1, y in loader:
            _ = model([m0.to(device), m1.to(device)])
            r0.append(model.reps[0].mean(dim=1).cpu())
            r1.append(model.reps[1].mean(dim=1).cpu())
            tgts.append(y.cpu())
    X = {"modality0": torch.cat(r0).float(), "modality1": torch.cat(r1).float()}
    return X, torch.cat(tgts).float()


# ═══════════════════════════════════════════════════════════════════════════════
# PID UTILITIES (same as ccs_source_aware_base.py)
# ═══════════════════════════════════════════════════════════════════════════════

def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):
    probs = torch.clamp(probs, min=eps, max=1.0)
    ce  = -torch.log(probs)[torch.arange(len(targets)), targets.long()].mean()
    acc = (probs.argmax(dim=1) == targets).float().mean()
    return acc.item(), ce.item()


def compute_log_py(targets, num_classes):
    counts = torch.bincount(targets, minlength=num_classes).float()
    probs  = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return torch.log(probs)[targets]


def ce_per_sample(targets, probs, eps=1e-12):
    probs = torch.clamp(probs, eps, 1.0)
    return -torch.log(probs[torch.arange(len(targets)), targets])


def compute_probs_and_ce(logits_list, targets):
    probs_list = [F.softmax(l, dim=1) for l in logits_list]
    ce_list    = [ce_per_sample(targets, p) for p in probs_list]
    return probs_list, ce_list


def compute_ccs(ce_list, log_py):
    i_list    = [-ce - log_py for ce in ce_list]
    same_sign = torch.sign(i_list[0]) == torch.sign(i_list[1])
    worst_ce  = torch.max(torch.stack(ce_list, dim=1), dim=1).values
    return torch.where(same_sign, worst_ce, -log_py)


def logp(p):
    return torch.log(torch.clamp(p, 1e-12, 1.0))


def compute_entropy_from_targets(targets, num_classes):
    targets = torch.as_tensor(targets).long()
    counts  = torch.bincount(targets, minlength=num_classes).float()
    probs   = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return -(probs * torch.log(probs)).sum().item()


def compute_pid_global(joint_ce, mod0_ce, mod1_ce, red_ce, src_red_ce,
                       targets, num_classes):
    hy = compute_entropy_from_targets(targets, num_classes)
    print(f"H(Y)={hy:.4f}  joint={joint_ce:.4f}  red={red_ce:.4f}  "
          f"src_red={src_red_ce:.4f}  mod0={mod0_ce:.4f}  mod1={mod1_ce:.4f}")

    mod0_ce    = min(mod0_ce, hy);    mod1_ce    = min(mod1_ce, hy)
    red_ce     = min(red_ce,  hy);    src_red_ce = min(src_red_ce, hy)
    red_ce     = max(red_ce,     joint_ce, mod0_ce, mod1_ce)
    src_red_ce = max(src_red_ce, joint_ce, mod0_ce, mod1_ce)
    red_ce     = min(red_ce, src_red_ce)
    mod0_ce    = min(max(mod0_ce, joint_ce), red_ce)
    mod1_ce    = min(max(mod1_ce, joint_ce), red_ce)

    i_total = hy - joint_ce
    i_r     = hy - red_ce;    i_r_src = hy - src_red_ce
    i_u0    = (hy - mod0_ce) - i_r
    i_u1    = (hy - mod1_ce) - i_r
    i_s     = i_total - i_u0 - i_u1 - i_r

    if i_s < 0:
        i_r -= i_s; i_r_src -= i_s
        i_u0 = (hy - mod0_ce) - i_r
        i_u1 = (hy - mod1_ce) - i_r
        i_s  = 0.0

    ratio = i_r_src / (i_r + 1e-10)
    print(f"R={i_r:.4f} ({100*ratio:.1f}% Source)  "
          f"U_mod0={i_u0:.4f}  U_mod1={i_u1:.4f}  S={i_s:.4f}  I={i_total:.4f}")


def compute_pointwise_pid_with_source(d, num_classes):
    targets = d["true_labels"].long()
    log_py  = compute_log_py(targets, num_classes)
    pid_list = []

    for j, m0, m1, npr, r_src, y, lpy in zip(
        d["pred_joint"], d["pred_modalities"][0], d["pred_modalities"][1],
        d["redundancy_pointwise_ce"], d["source_redundancy_preds"], targets, log_py
    ):
        y        = y.long()
        joint_ce = -logp(F.softmax(j,     dim=0))[y]
        mod0_ce  = -logp(F.softmax(m0,    dim=0))[y]
        mod1_ce  = -logp(F.softmax(m1,    dim=0))[y]
        red_ce   = npr
        src_ce   = -logp(F.softmax(r_src, dim=0))[y]
        hy       = -lpy

        joint_ce = min(red_ce,  joint_ce)
        src_ce   = min(red_ce,  src_ce)
        red_ce   = min(red_ce,  hy)
        src_ce   = min(src_ce,  hy)
        mod0_ce  = max(mod0_ce, joint_ce)
        mod1_ce  = max(mod1_ce, joint_ce)

        total = hy - joint_ce
        r_val = max(hy - red_ce, hy - src_ce)
        u0    = hy - mod0_ce - r_val
        u1    = hy - mod1_ce - r_val
        s     = total - u0 - u1 - r_val
        pid_list.append([u0, u1, r_val, s])

    return np.array(pid_list)


def normalize_pid(pid):
    pid_ = np.maximum(pid, 0)
    pid_ /= pid_.sum(axis=1, keepdims=True) + 1e-12
    return pid_


def compute_redundancy_metrics(y_pred_dict):
    results = {}
    for key in ["modality0", "modality1", "average"]:
        acc, ce = traditional_cross_entropy_from_probs(
            softmax(y_pred_dict[key]), y_pred_dict["targets"])
        results[key] = {"accuracy": acc, "cross_entropy": ce}
    return results


def print_model_metrics(d):
    print(f"{'Model':<12s} | {'Acc':>8s} | {'CE':>8s}")
    print("-" * 36)
    print(f"{'Joint':<12s} | {d['joint_acc']:8.4f} | {d['joint_ce']:8.4f}")
    print(f"{'Modality 0':<12s} | {d['modalities_acc'][0]:8.4f} | {d['modalities_ce'][0]:8.4f}")
    print(f"{'Modality 1':<12s} | {d['modalities_acc'][1]:8.4f} | {d['modalities_ce'][1]:8.4f}")


def print_redundancy_metrics(results):
    for key, name in [("modality0", "Red Mod 0"), ("modality1", "Red Mod 1"),
                      ("average",   "Red Joint")]:
        print(f"{name:<14s} | {results[key]['accuracy']:8.4f} | {results[key]['cross_entropy']:8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    case = ("Synergy (biased)" if args.biased
            else "Redundancy+Uniqueness (unbiased)")
    print(f"\n{'='*60}")
    print(f"TriFeatures CoMM CCS-PID")
    print(f"  Case     : {case}")
    print(f"  Task     : {args.task}  ({NUM_CLASSES} classes)")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    # ========= 1. DATA =========
    print("Loading data…")
    traindata, testdata = get_dataloaders(
        args.data_root, args.task, args.biased, args.bs, args.num_workers)
    print(f"  train={len(traindata.dataset)}  test={len(testdata.dataset)}")

    # ========= 2. MODEL =========
    model = TriFeaturesCoMMModel(
        num_classes=NUM_CLASSES,
        embed_dim=args.embed_dim,
        cnn_channels=args.cnn_channels,
        seq_len=16,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ========= 3. TRAIN =========
    print("\nTraining…")
    model = train_comm(model, traindata, testdata,
                       epochs=args.epochs, lr=args.lr,
                       weight_decay=args.weight_decay,
                       save=args.saved_model)

    # ========= 4. TEST =========
    print("\nTest performance:")
    d = test_comm(model, testdata)
    print_model_metrics(d)

    # ========= 5. EXTRACT REPRESENTATIONS =========
    print("\nExtracting representations…")
    X_train, y_train = extract_split(model, traindata)
    X_test,  y_test  = extract_split(model, testdata)

    # ========= 6. CCS REDUNDANCY =========
    targets = d["true_labels"].long()
    log_py  = compute_log_py(targets, NUM_CLASSES)
    _, ce_list = compute_probs_and_ce(d["pred_modalities"], targets)
    ccs = compute_ccs(ce_list, log_py)
    d["redundancy_ce"]           = ccs.mean().item()
    d["redundancy_pointwise_ce"] = ccs.numpy()

    # ========= 7. SOURCE REDUNDANCY =========
    y_pred_dict = return_redundancy_test_performances(
        X_train, X_test, X_test,
        y_train.long(), y_test.long(), y_test.long(),
        f"trifeatures_{args.task}_biased={args.biased}",
        distribution_target="categorical",
        num_classes=NUM_CLASSES,
        lr=1e-5
    )

    results = compute_redundancy_metrics(y_pred_dict)
    print_redundancy_metrics(results)

    d["source_redundancy_pointwise_ce"] = results["average"]["cross_entropy"]
    d["source_redundancy_preds"]        = y_pred_dict["average"]

    # ========= 8. GLOBAL PID =========
    print("\nGlobal PID:")
    compute_pid_global(
        d["joint_ce"], d["modalities_ce"][0], d["modalities_ce"][1],
        d["redundancy_ce"], d["source_redundancy_pointwise_ce"],
        targets=d["true_labels"], num_classes=NUM_CLASSES,
    )

    # ========= 9. POINTWISE PID =========
    pid_source = compute_pointwise_pid_with_source(d, NUM_CLASSES)

    print(f"\nMean pointwise PID [U_mod0, U_mod1, R, S]:")
    print(np.mean(pid_source, axis=0))
    pid_norm = normalize_pid(pid_source)
    print("Normalised mean:", np.mean(pid_norm, axis=0))

    # Correction loop (distribution level)
    pid_corrected = []
    for pid_i in pid_source:
        pid_i = list(pid_i)
        if pid_i[0] < 0 and pid_i[1] >= 0:
            pid_i[3] += pid_i[0]; pid_i[0] = 0
        elif pid_i[1] < 0 and pid_i[0] >= 0:
            pid_i[3] += pid_i[1]; pid_i[1] = 0
        pid_corrected.append(pid_i)
    pid_source = np.array(pid_corrected)

    print(f"\nAfter Correction Mean pointwise PID [U_mod0, U_mod1, R, S]:")
    print(np.mean(pid_source, axis=0))
    pid_norm = normalize_pid(pid_source)
    print("After Correction Normalised mean:", np.mean(pid_norm, axis=0))
