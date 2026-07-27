"""
CoMM architecture + LSMI pointwise PID estimator on the TriFeatures bimodal dataset.

Pipeline:
  1. Train CoMM encoders (AlexNetEncoder → PatchedInputAdapter → FusionTransformer)
     with cross-entropy on the chosen task.
  2. Extract mean-pooled token representations: z_i = mean_pool(adapter(encoder(x_i)))
     yielding (N, embed_dim) vectors for each modality.
  3. Train LSMI discriminators p(y|z1), p(y|z2), p(y|z1,z2) on those representations.
  4. Train MargKernel entropy estimators H(Z1), H(Z2).
  5. Compute LSMI PID → per-sample R, U1, U2, S.
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

from utils_lsmi import MargKernel, cls_network, feature_dataset, setup_seed

multiprocessing.set_start_method('fork', force=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
# Data / task
parser.add_argument("--data-root",    default="/home/rlouiset/trifeatures_data", type=str)
parser.add_argument("--task",         default="share",
                    choices=["share", "unique1", "unique2", "synergy"],
                    help=("share: classify shared shape (redundancy). "
                          "unique1/unique2: classify texture unique to modality 1 or 2. "
                          "synergy: binary label from correlated texture×color pair."))
parser.add_argument("--biased",       action="store_true")
parser.add_argument("--num-classes",  default=10,  type=int,
                    help="10 for share/unique tasks, 2 for synergy")
# Model training
parser.add_argument("--bs",           default=64,  type=int)
parser.add_argument("--num-workers",  default=4,   type=int)
parser.add_argument("--embed-dim",    default=512, type=int)
parser.add_argument("--epochs",       default=50,  type=int)
parser.add_argument("--lr",           default=1e-4, type=float)
parser.add_argument("--weight-decay", default=1e-2, type=float)
parser.add_argument("--saved-model",  default=None, type=str)
# LSMI
parser.add_argument("--embed-size",     default=128, type=int,
                    help="Hidden dim of LSMI discriminators")
parser.add_argument("--lsmi-bs",        default=256, type=int)
parser.add_argument("--epochs-disc",    default=30,  type=int)
parser.add_argument("--epochs-entropy", default=30,  type=int)
parser.add_argument("--seed",           default=42,  type=int)
args = parser.parse_args()

setup_seed(args.seed)
NUM_CLASSES = args.num_classes


# ═══════════════════════════════════════════════════════════════════════════════
# TRIFEATURES DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class Trifeatures(Dataset):
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
        self.root                = root
        self.split               = split
        self.num_per_combination = num_per_combination
        self.transform           = transform
        self.split_ratio         = 0.8
        self.rng                 = np.random.default_rng(seed)
        self._base_templates     = {s: self._render_plain_shape(s)
                                    for s in self.BASE_SHAPES}
        if not self._check_integrity():
            self.generate_data()
        self.root           = os.path.join(self.root, self.split)
        self.name_images    = sorted(f for f in os.listdir(self.root) if f.endswith('.png'))
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
        parts = re.split(r'_', filename.split('.')[0])
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
    def __init__(self, root, split="train", task="share",
                 share_attr="shape", unique_attr="texture",
                 synergy_attr=("texture", "color"),
                 biased=False, max_size=10000,
                 num_per_combination=3, transform=None, seed=42):
        super().__init__(root, split, num_per_combination=num_per_combination,
                         transform=transform, seed=seed)
        self.task         = task
        self.share_attr   = share_attr
        self.unique_attr  = unique_attr
        self.synergy_attr = synergy_attr
        self.biased       = biased
        self.max_size     = int(max_size)

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

        sv0 = targets[:, a2i[self.synergy_attr[0]]]
        sv1 = targets[:, a2i[self.synergy_attr[1]]]
        synergy_eq = np.zeros((n, n), dtype=bool)
        for p in self.correlated_feature_pairs:
            m0 = (sv0.reshape(n, 1) == p[0]).repeat(n, axis=1)
            m1 = (sv1.reshape(1, n) == p[1]).repeat(n, axis=0)
            synergy_eq |= (m0 & m1)

        if self.task == "synergy":
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

    def reshuffle(self):
        """Re-sample random pairs for the next epoch (data augmentation)."""
        self.idx_pairs = self._get_idx_pairs()

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
# CoMM ARCHITECTURE  (AlexNetEncoder → PatchedInputAdapter → FusionTransformer)
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


class AlexNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import alexnet
        self.features = alexnet(weights=None).features

    def forward(self, x):
        return self.features(x)   # (B, 256, 6, 6)


class PatchedInputAdapter(nn.Module):
    def __init__(self, num_channels=256, dim_tokens=512, image_size=6):
        super().__init__()
        self.proj = nn.Conv2d(num_channels, dim_tokens, kernel_size=1, stride=1)
        pos = _build_2d_sincos_posemb(image_size, image_size, dim_tokens)
        self.register_buffer('pos_emb', pos)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = self.proj(x)
        pos    = F.interpolate(self.pos_emb, size=(H, W),
                               mode='bicubic', align_corners=False)
        return (tokens + pos).flatten(2).transpose(1, 2)  # (B, H*W, dim_tokens)


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


class TriFeaturesCoMMModel(nn.Module):
    def __init__(self, num_classes, embed_dim=512):
        super().__init__()
        self.encoders = nn.ModuleList([AlexNetEncoder(), AlexNetEncoder()])
        self.adapters = nn.ModuleList([
            PatchedInputAdapter(num_channels=256, dim_tokens=embed_dim, image_size=6),
            PatchedInputAdapter(num_channels=256, dim_tokens=embed_dim, image_size=6),
        ])
        self.fusion    = FusionTransformer(width=embed_dim, n_heads=8, n_layers=1)
        self.head      = nn.Linear(embed_dim, num_classes)
        self.mod_heads = nn.ModuleList([nn.Linear(embed_dim, num_classes),
                                        nn.Linear(embed_dim, num_classes)])
        self.reps = []

    def forward(self, inputs):
        seqs = [adapter(enc(x.float()))
                for enc, adapter, x in zip(self.encoders, self.adapters, inputs)]
        self.reps    = seqs
        fused        = self.fusion(seqs)
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
        if hasattr(traindata.dataset, 'reshuffle'):
            traindata.dataset.reshuffle()
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
                vn += len(y)
                vc += (joint.argmax(1) == y).sum().item()
        val_loss /= vn
        print(f"Epoch {epoch}  train={total_loss/n:.4f}  val={val_loss:.4f}  acc={vc/vn:.4f}")
        if val_loss < best_val:
            best_val   = val_loss
            best_model = copy.deepcopy(model)
            print("  → best")
            if save:
                torch.save(model, save)

    return best_model


def extract_representations(model, loader):
    """
    Extract per-modality representations as mean-pooled adapter sequences.
    Returns (N, embed_dim) tensors for each modality.
    """
    model.eval()
    r0, r1, tgts = [], [], []
    with torch.no_grad():
        for m0, m1, y in loader:
            _ = model([m0.to(device), m1.to(device)])
            r0.append(model.reps[0].mean(dim=1).cpu())   # (B, 36, D) → (B, D)
            r1.append(model.reps[1].mean(dim=1).cpu())
            tgts.append(y.cpu())
    return torch.cat(r0).float(), torch.cat(r1).float(), torch.cat(tgts).long()


# ═══════════════════════════════════════════════════════════════════════════════
# LSMI ESTIMATORS
# ═══════════════════════════════════════════════════════════════════════════════

def obtain_discriminator(train_loader, repr_dim, embed_size, n_classes, n_epochs):
    """Train p(y|z1), p(y|z2), p(y|z1,z2) classifiers on latent representations."""
    d1 = cls_network(repr_dim,     embed_size, n_classes).to(device)
    d2 = cls_network(repr_dim,     embed_size, n_classes).to(device)
    dj = cls_network(repr_dim * 2, embed_size, n_classes).to(device)
    models    = [d1, d2, dj]
    optimizer = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        losses = 0.0; num_samples = 0
        for batch in train_loader:
            z1, z2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            loss = (criterion(d1(z1), y) + criterion(d2(z2), y)
                    + criterion(dj(torch.cat([z1, z2], dim=1)), y))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses += loss.item() * len(y); num_samples += len(y)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  [disc]    Epoch [{epoch+1}/{n_epochs}]  Loss: {losses/num_samples:.4f}')
    return models


def obtain_entropy_estimator(train_loader, repr_dim, n_epochs):
    """Train MargKernel models for H(Z1) and H(Z2)."""
    mk1 = MargKernel(repr_dim).to(device)
    mk2 = MargKernel(repr_dim).to(device)
    models    = [mk1, mk2]
    optimizer = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(n_epochs):
        for m in models: m.train()
        losses = 0.0; n = 0
        for batch in train_loader:
            z1, z2 = batch[0].to(device), batch[1].to(device)
            loss = mk1(z1) + mk2(z2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses += loss.item() * len(z1); n += len(z1)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  [entropy] Epoch [{epoch+1}/{n_epochs}]  Loss: {losses/n:.4f}')
    return models


def get_mutual_info(loader, model, modality, n_classes):
    """I(Zi; Y) ≈ E[ log p(y|zi) ] + log K."""
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            z1, z2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            if modality == 'modality_1':
                x = z1
            elif modality == 'modality_2':
                x = z2
            else:
                x = torch.cat([z1, z2], dim=1)
            rows = torch.arange(x.size(0), device=device)
            out  = model(x)
            info.append(math.log(n_classes) + F.log_softmax(out, dim=1)[rows, y])
    return torch.cat(info).detach()


def get_entropy(loader, model, modality):
    """H(Zi) via MargKernel."""
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            x = (batch[0] if modality == 'modality_1' else batch[1]).to(device)
            info.append(model(x))
    return torch.cat(info).detach()


def RUS_adjustment(rus):
    r_orig, u_1_orig, u_2_orig, s_orig = rus
    R_mean  = r_orig.detach().mean()
    U1_mean = u_1_orig.detach().mean()
    U2_mean = u_2_orig.detach().mean()
    S_mean  = s_orig.detach().mean()
    adj = torch.tensor(0.0, dtype=R_mean.dtype, device=R_mean.device)
    if R_mean < 0 or S_mean < 0:
        adj = -torch.min(R_mean, S_mean)
    elif U1_mean < 0 or U2_mean < 0:
        adj = torch.min(U1_mean, U2_mean)
    return r_orig + adj, u_1_orig - adj, u_2_orig - adj, s_orig + adj


def normalize_pid(pid):
    pid_ = np.maximum(pid, 0)
    pid_ /= pid_.sum(axis=1, keepdims=True) + 1e-12
    return pid_


def LSMI_estimation(loader, discriminator, entropy_estimator, n_classes):
    I1Y  = get_mutual_info(loader, discriminator[0], 'modality_1',  n_classes)
    I2Y  = get_mutual_info(loader, discriminator[1], 'modality_2',  n_classes)
    I12Y = get_mutual_info(loader, discriminator[2], 'modality_12', n_classes)
    H1   = get_entropy(loader, entropy_estimator[0], 'modality_1')
    H2   = get_entropy(loader, entropy_estimator[1], 'modality_2')

    r_plus  = torch.minimum(H1, H2)
    r_minus = torch.minimum(H1 - I1Y, H2 - I2Y)
    r  = r_plus - r_minus
    u1 = I1Y  - r
    u2 = I2Y  - r
    s  = I12Y - r - u1 - u2

    r_adj, u1_adj, u2_adj, s_adj = RUS_adjustment([r, u1, u2, s])
    print(f"  R={r_adj.mean():.4f}  U1={u1_adj.mean():.4f}  "
          f"U2={u2_adj.mean():.4f}  S={s_adj.mean():.4f}")

    return r.cpu().numpy(), u1.cpu().numpy(), u2.cpu().numpy(), s.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_entropy_from_targets(targets, num_classes, verbose=True):
    targets = torch.as_tensor(targets).long()
    counts  = torch.bincount(targets, minlength=num_classes).float()
    probs   = counts / counts.sum()
    if verbose:
        print("counts     :", counts.tolist())
        print("p(y) per-cls:", [round(x, 4) for x in probs.tolist()])
        print("sum p(y)    :", probs.sum().item())
        print("n_samples   :", int(counts.sum().item()))
        print("max label   :", int(targets.max().item()),
              " min label:", int(targets.min().item()))
    probs = torch.clamp(probs, 1e-12, 1.0)
    return -(probs * torch.log(probs)).sum().item()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    case = "Synergy (biased)" if args.biased else "Redundancy+Uniqueness (unbiased)"
    print(f"\n{'='*60}")
    print(f"TriFeatures CoMM LSMI-PID")
    print(f"  Case     : {case}")
    print(f"  Task     : {args.task}  ({NUM_CLASSES} classes)")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    # ========= 1. DATA =========
    print("Loading data…")
    traindata, testdata = get_dataloaders(
        args.data_root, args.task, args.biased, args.bs, args.num_workers)
    print(f"  train={len(traindata.dataset)}  test={len(testdata.dataset)}")

    # ========= 1b. LABEL DISTRIBUTION SANITY CHECK =========
    # Same check as trifeatures_comm_ccs_pid.py's compute_entropy_from_targets(verbose=True) --
    # compare counts/p(y)/n_samples/H(Y) here against the CCS run's printout to confirm
    # both scripts are seeing the same underlying test-set label distribution.
    print("\nTest-set label distribution (before training):")
    test_labels = torch.cat([y for _, _, y in testdata])
    H_Y_check = compute_entropy_from_targets(test_labels, NUM_CLASSES, verbose=True)
    print(f"H(Y) [test, pre-training] = {H_Y_check:.4f} nats\n")

    # ========= 2. MODEL =========
    model = TriFeaturesCoMMModel(
        num_classes=NUM_CLASSES, embed_dim=args.embed_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ========= 3. TRAIN =========
    print("\nTraining CoMM…")
    model = train_comm(model, traindata, testdata,
                       epochs=args.epochs, lr=args.lr,
                       weight_decay=args.weight_decay,
                       save=args.saved_model)

    # ========= 4. EXTRACT REPRESENTATIONS =========
    print("\nExtracting representations…")
    train_z1, train_z2, y_train = extract_representations(model, traindata)
    test_z1,  test_z2,  y_test  = extract_representations(model, testdata)
    repr_dim = train_z1.shape[1]
    print(f"  Repr dim: {repr_dim}  |  train N={len(y_train)}  test N={len(y_test)}")

    # ========= 5. BUILD LSMI LOADERS =========
    lsmi_train = DataLoader(
        feature_dataset(train_z1, train_z2, y_train),
        batch_size=args.lsmi_bs, shuffle=True,  num_workers=0)
    lsmi_test  = DataLoader(
        feature_dataset(test_z1,  test_z2,  y_test),
        batch_size=args.lsmi_bs, shuffle=False, num_workers=0)

    # ========= 6. TRAIN LSMI ESTIMATORS =========
    print("\nTraining discriminators  (p(y|z1), p(y|z2), p(y|z1,z2))…")
    discriminator = obtain_discriminator(
        lsmi_train, repr_dim, args.embed_size, NUM_CLASSES, args.epochs_disc)

    print("\nTraining entropy estimators  (H(Z1), H(Z2))…")
    entropy_estimator = obtain_entropy_estimator(
        lsmi_train, repr_dim, args.epochs_entropy)

    # ========= 7. LSMI PID =========
    print("\nTrain PID:")
    LSMI_estimation(lsmi_train, discriminator, entropy_estimator, NUM_CLASSES)

    print("\nTest PID:")
    r, u1, u2, s = LSMI_estimation(lsmi_test, discriminator, entropy_estimator, NUM_CLASSES)

    # ========= 8. RAW / ADJUSTED / NORMALISED =========
    pid = np.stack([u1, u2, r, s], axis=1)   # columns: [U1, U2, R, S]

    print("\nBefore adjustment — mean [U1, U2, R, S]:")
    print(np.mean(pid, axis=0))

    r_adj, u1_adj, u2_adj, s_adj = RUS_adjustment(
        [torch.tensor(r), torch.tensor(u1),
         torch.tensor(u2), torch.tensor(s)])
    r_adj   = r_adj.numpy();  u1_adj = u1_adj.numpy()
    u2_adj  = u2_adj.numpy(); s_adj  = s_adj.numpy()

    print("\nAfter adjustment — mean [U1, U2, R, S]:")
    print(np.mean(np.stack([u1_adj, u2_adj, r_adj, s_adj], axis=1), axis=0))

    pid_norm = normalize_pid(pid)
    print("\nNormalised mean [U1, U2, R, S]:")
    print(np.mean(pid_norm, axis=0))
