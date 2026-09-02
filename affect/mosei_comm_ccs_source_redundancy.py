"""
CoMM/FactorCL architecture for MOSEI PID estimation.
  - Per-modality encoder : Conv1d projection → sinusoidal pos-emb → 5-layer Transformer (5 heads)
  - Fusion               : CLS token + concat sequences → 1-layer Transformer (8 heads) → CLS output
Reference: Dufumier et al. 2025 (https://arxiv.org/pdf/2409.07402)
           Liang et al. 2023  (FactorCL)
"""
import math
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pickle

from torch.utils.data import Dataset, DataLoader

from utils_ours import return_redundancy_test_performances

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── Args ────────────────────────────────────────────────────────────────────

# /Users/robinlouiset/Downloads/mosei_senti_data.pkl
# /lustre/fswork/projects/rech/haj/uik24xv/local/mosei_senti_data.pkl

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/home/rlouiset/mosei_senti_data.pkl", type=str)
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--embed-dim", default=40, type=int,
                    help="Transformer embedding dim (40 as in CoMM/FactorCL)")
parser.add_argument("--num-classes", default=2, type=int,
                    help="2=binary (pos/neg), 7=7-class sentiment")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--weight-decay", default=1e-2, type=float)
parser.add_argument("--seq-len", default=50, type=int,
                    help="Sequence length (50 for MOSEI, 20 for UR-FUNNY)")
parser.add_argument("--pos-enc", action="store_true",
                    help="Add sinusoidal positional encoding (off by default, as in CoMM/MOSI)")
parser.add_argument("--noise-std", default=0.1, type=float)
parser.add_argument("--temporal-dropout-max", default=0.8, type=float)
parser.add_argument("--mod0", default="vision", choices=["vision", "audio", "text"])
parser.add_argument("--mod1", default="text",   choices=["vision", "audio", "text"])
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()

assert args.mod0 != args.mod1, "--mod0 and --mod1 must be different"

softmax = torch.nn.Softmax(dim=-1)

MODALITY_DIMS = {"vision": 35, "audio": 74, "text": 300}


# ─── Data loading ─────────────────────────────────────────────────────────────

def binarize(labels):
    return (labels >= 0).astype(np.int64)

def to_7class(labels):
    return np.clip(np.floor(labels + 3.5).astype(np.int64), 0, 6)


class MOSEIDataset(Dataset):
    def __init__(self, split_dict, mod0_key="vision", mod1_key="text",
                 num_classes=2, augment=False, noise_std=0.1, temporal_dropout_max=0.8):
        self.mod0 = torch.from_numpy(split_dict[mod0_key].astype(np.float32))
        self.mod1 = torch.from_numpy(split_dict[mod1_key].astype(np.float32))
        raw_labels = split_dict["labels"].squeeze()
        labels = binarize(raw_labels) if num_classes == 2 else to_7class(raw_labels)
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment
        self.noise_std = noise_std
        self.temporal_dropout_max = temporal_dropout_max

    def _augment(self, x):
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.temporal_dropout_max > 0:
            rate = float(torch.empty(1).uniform_(0, self.temporal_dropout_max))
            mask = torch.rand(x.size(0)) > rate
            x = x * mask.unsqueeze(1).float()
        return x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        m0, m1 = self.mod0[idx], self.mod1[idx]
        if self.augment:
            m0, m1 = self._augment(m0), self._augment(m1)
        return m0, m1, self.labels[idx]


def get_dataloaders(path, mod0_key, mod1_key, num_classes=2,
                    batch_size=32, num_workers=4, noise_std=0.1, temporal_dropout_max=0.8):
    with open(path, "rb") as f:
        data = pickle.load(f)
    train_ds = MOSEIDataset(data["train"], mod0_key, mod1_key, num_classes,
                            augment=True, noise_std=noise_std,
                            temporal_dropout_max=temporal_dropout_max)
    valid_ds = MOSEIDataset(data["valid"], mod0_key, mod1_key, num_classes, augment=False)
    test_ds  = MOSEIDataset(data["test"],  mod0_key, mod1_key, num_classes, augment=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# ─── CoMM Architecture ───────────────────────────────────────────────────────

def build_sincos_posemb(max_seq_len, embed_dim):
    """Sinusoidal positional embedding (1, max_seq_len, embed_dim), non-learnable."""
    assert embed_dim % 2 == 0, "embed_dim must be even for sinusoidal encoding"
    pe = torch.zeros(max_seq_len, embed_dim)
    pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, max_seq_len, embed_dim)


class CoMMEncoder(nn.Module):
    """
    Per-modality encoder (CoMM/FactorCL):
      Conv1d(n_features → embed_dim) + optional sinusoidal pos-emb
      + 5-layer TransformerEncoder (5 heads, pre-norm).
    Returns token sequence  (batch, T, embed_dim).
    """
    def __init__(self, n_features, embed_dim=40, max_seq_len=50,
                 n_heads=5, n_layers=5, positional_encoding=False):
        super().__init__()
        self.conv = nn.Conv1d(n_features, embed_dim, kernel_size=1, bias=False)
        self.use_pos_enc = positional_encoding
        if positional_encoding:
            self.register_buffer("pos_emb", build_sincos_posemb(max_seq_len, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        # x: (batch, T, n_features)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # → (batch, T, embed_dim)
        if self.use_pos_enc:
            x = x + self.pos_emb[:, :x.size(1)]
        return self.transformer(x)                           # (batch, T, embed_dim)


class FusionTransformer(nn.Module):
    """
    CoMM fusion module:
      prepend CLS token → concat modality sequences → 1-layer self-attention → CLS output.
    """
    def __init__(self, embed_dim=40, n_heads=8, n_layers=1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, sequences):
        # sequences: list of (batch, T_i, embed_dim)
        batch = sequences[0].size(0)
        cls = self.cls_token.expand(batch, -1, -1)       # (batch, 1, embed_dim)
        tokens = torch.cat([cls] + sequences, dim=1)     # (batch, 1 + ΣT_i, embed_dim)
        out = self.transformer(tokens)
        return self.norm(out[:, 0])                      # (batch, embed_dim)  — CLS token


class CoMMMultimodalModel(nn.Module):
    """
    Full supervised CoMM model.
    forward() returns (joint_logits, [seq0, seq1], [mod0_logits, mod1_logits])
    matching the RedundancyAwareMMDL interface so extract_representations() works unchanged.
    Per-modality heads use mean-pooled sequences.
    """
    def __init__(self, mod0_dim, mod1_dim, num_classes,
                 embed_dim=40, seq_len=50, positional_encoding=False):
        super().__init__()
        self.encoders = nn.ModuleList([
            CoMMEncoder(mod0_dim, embed_dim, seq_len, positional_encoding=positional_encoding),
            CoMMEncoder(mod1_dim, embed_dim, seq_len, positional_encoding=positional_encoding),
        ])
        self.fusion  = FusionTransformer(embed_dim)
        self.head    = nn.Linear(embed_dim, num_classes)
        self.mod_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes),
            nn.Linear(embed_dim, num_classes),
        ])
        self.reps    = []
        self.fuseout = None

    def forward(self, inputs):
        seqs = [enc(x.float()) for enc, x in zip(self.encoders, inputs)]
        self.reps    = seqs
        fused        = self.fusion(seqs)
        self.fuseout = fused
        joint_logits = self.head(fused)
        mod_logits   = [h(s.mean(dim=1)) for h, s in zip(self.mod_heads, seqs)]
        return joint_logits, seqs, mod_logits


# ─── Training loop ───────────────────────────────────────────────────────────

def accuracy(true, pred):
    return (pred == true).float().mean().item()


def train_comm(model, traindata, validdata, epochs, lr, weight_decay, save=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_model   = copy.deepcopy(model)
    best_valloss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; n = 0
        for batch in traindata:
            m0, m1, y = batch
            inputs  = [m0.to(device), m1.to(device)]
            y = y.to(device).long()
            joint, _, mod_logits = model(inputs)
            loss = criterion(joint, y)
            for logits in mod_logits:
                loss = loss + criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()
            total_loss += loss.item() * len(y); n += len(y)

        model.eval()
        val_loss = 0.0; val_n = 0; val_correct = 0
        with torch.no_grad():
            for batch in validdata:
                m0, m1, y = batch
                inputs = [m0.to(device), m1.to(device)]
                y = y.to(device).long()
                joint, _, _ = model(inputs)
                val_loss    += criterion(joint, y).item() * len(y)
                val_n       += len(y)
                val_correct += (joint.argmax(1) == y).sum().item()

        val_loss /= val_n
        val_acc   = val_correct / val_n
        print(f"Epoch {epoch}  train={total_loss/n:.4f}  val={val_loss:.4f}  acc={val_acc:.4f}")

        if val_loss < best_valloss:
            best_valloss = val_loss
            best_model   = copy.deepcopy(model)
            print("  -> Best")
            if save:
                torch.save(model, save)

    return best_model


def test_comm(model, testdata, num_classes):
    """Evaluate model, return the same dict as single_test() for PID compatibility."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    pred_joint, pred_m0, pred_m1, true_labels = [], [], [], []
    total_joint = 0.0; total_m0 = 0.0; total_m1 = 0.0; n = 0

    with torch.no_grad():
        for batch in testdata:
            m0, m1, y = batch
            inputs = [m0.to(device), m1.to(device)]
            y = y.to(device).long()
            joint, _, mod_logits = model(inputs)

            total_joint += criterion(joint,        y).item() * len(y)
            total_m0    += criterion(mod_logits[0], y).item() * len(y)
            total_m1    += criterion(mod_logits[1], y).item() * len(y)
            n           += len(y)

            pred_joint.append(joint.cpu())
            pred_m0.append(mod_logits[0].cpu())
            pred_m1.append(mod_logits[1].cpu())
            true_labels.append(y.cpu())

    pred_joint  = torch.cat(pred_joint)
    pred_m0     = torch.cat(pred_m0)
    pred_m1     = torch.cat(pred_m1)
    true_labels = torch.cat(true_labels)

    return {
        "joint_acc":      accuracy(true_labels, pred_joint.argmax(1)),
        "modalities_acc": [accuracy(true_labels, pred_m0.argmax(1)),
                           accuracy(true_labels, pred_m1.argmax(1))],
        "joint_ce":      total_joint / n,
        "modalities_ce": [total_m0 / n, total_m1 / n],
        "pred_joint":     pred_joint,
        "true_labels":    true_labels,
        "pred_modalities": [pred_m0, pred_m1],
    }


# ─── Representation extraction ────────────────────────────────────────────────

def extract_representations(model, dataloader):
    model.eval()
    reps0, reps1, targets = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            m0, m1, y = batch
            _ = model([m0.to(device), m1.to(device)])
            # model.reps are sequences (batch, T, embed_dim) → mean-pool for PID
            reps0.append(model.reps[0].mean(dim=1).cpu())
            reps1.append(model.reps[1].mean(dim=1).cpu())
            targets.append(y.cpu())
    return {
        "modality0": torch.cat(reps0),
        "modality1": torch.cat(reps1),
        "targets":   torch.cat(targets),
    }


def extract_split(model, loader):
    d = extract_representations(model, loader)
    X = {"modality0": d["modality0"].float(), "modality1": d["modality1"].float()}
    return X, d["targets"].float()


# ─── Metric helpers ───────────────────────────────────────────────────────────

def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):
    probs = torch.clamp(probs, min=eps, max=1.0)
    ce  = -torch.log(probs)[torch.arange(len(targets)), targets.long()].mean()
    acc = (probs.argmax(dim=1) == targets).float().mean()
    return acc.item(), ce.item()


def print_model_metrics(d, mod0_name="mod0", mod1_name="mod1"):
    print(f"{'Model':<12s} | {'Acc':>8s} | {'CE':>8s}")
    print("-" * 36)
    print(f"{'Joint':<12s} | {d['joint_acc']:8.4f} | {d['joint_ce']:8.4f}")
    print(f"{mod0_name:<12s} | {d['modalities_acc'][0]:8.4f} | {d['modalities_ce'][0]:8.4f}")
    print(f"{mod1_name:<12s} | {d['modalities_acc'][1]:8.4f} | {d['modalities_ce'][1]:8.4f}")


def print_redundancy_metrics(results, mod0_name="mod0", mod1_name="mod1"):
    mapping = {"modality0": f"Red {mod0_name}", "modality1": f"Red {mod1_name}", "average": "Red Joint"}
    for key, name in mapping.items():
        print(f"{name:<14s} | {results[key]['accuracy']:8.4f} | {results[key]['cross_entropy']:8.4f}")


# ─── PID utilities ────────────────────────────────────────────────────────────

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
    return -torch.sum(probs * torch.log(probs)).item()


def compute_pid_global(joint_ce, mod0_ce, mod1_ce, red_ce, src_red_ce,
                       num_classes, targets, mod0_name="mod0", mod1_name="mod1"):
    hy = compute_entropy_from_targets(targets, num_classes)
    print(f"H(Y)={hy:.4f}  joint={joint_ce:.4f}  red={red_ce:.4f}  "
          f"src_red={src_red_ce:.4f}  {mod0_name}={mod0_ce:.4f}  {mod1_name}={mod1_ce:.4f}")

    # ===== CLIPPING =====
    mod0_ce    = min(mod0_ce, hy)
    mod1_ce    = min(mod1_ce, hy)
    red_ce     = min(red_ce,  hy)
    src_red_ce = min(src_red_ce, hy)
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
        i_u0 = (hy - mod0_ce) - i_r; i_u1 = (hy - mod1_ce) - i_r; i_s = 0.0

    ratio = i_r_src / (i_r + 1e-10)
    print(f"R={i_r:.4f} ({100*ratio:.1f}% Source)  "
          f"U_{mod0_name}={i_u0:.4f}  U_{mod1_name}={i_u1:.4f}  S={i_s:.4f}  I={i_total:.4f}")


def compute_pointwise_pid_with_source(d, num_classes):
    targets = d["true_labels"].long()
    log_py  = compute_log_py(targets, num_classes)
    pid_list = []

    for j, m0, m1, npr, r_src, y, lpy in zip(
        d["pred_joint"], d["pred_modalities"][0], d["pred_modalities"][1],
        d["redundancy_pointwise_ce"], d["source_redundancy_preds"], targets, log_py,
    ):
        y        = y.long()
        joint_ce = -logp(F.softmax(j,     dim=0))[y]
        mod0_ce  = -logp(F.softmax(m0,    dim=0))[y]
        mod1_ce  = -logp(F.softmax(m1,    dim=0))[y]
        red_ce   = npr
        src_ce   = -logp(F.softmax(r_src, dim=0))[y]
        hy       = -lpy

        joint_ce = min(red_ce, joint_ce)
        #src_ce  = min(red_ce, src_ce)
        #red_ce   = min(red_ce, hy)
        #src_ce  = min(src_ce, hy)
        mod0_ce  = max(mod0_ce, joint_ce)
        mod1_ce = max(mod1_ce, joint_ce)

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


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    dim0 = MODALITY_DIMS[args.mod0]
    dim1 = MODALITY_DIMS[args.mod1]
    pair_name = f"{args.mod0}-{args.mod1}"
    print(f"Loading MOSEI  [{pair_name}]  dims=({dim0}, {dim1})  embed={args.embed_dim}")

    # ========= 1. LOAD DATA =========
    traindata, validdata, testdata = get_dataloaders(
        args.data_path, args.mod0, args.mod1,
        num_classes=args.num_classes,
        batch_size=args.bs, num_workers=args.num_workers,
        noise_std=args.noise_std, temporal_dropout_max=args.temporal_dropout_max,
    )

    # ========= 1b. DATASET STATISTICS =========
    def split_stats(loader, name):
        labels = loader.dataset.labels
        n = len(labels)
        counts = torch.bincount(labels, minlength=args.num_classes)
        probs  = counts.float() / n
        h      = -torch.sum(probs * torch.log(probs.clamp(1e-12))).item()
        class_str = "  ".join(f"c{i}:{probs[i]:.3f}" for i in range(args.num_classes))
        print(f"  {name:<6s}  n={n:>6d}  [{class_str}]  "
              f"H(Y)={h:.4f}  majority_acc={counts.max().item()/n:.4f}")

    total = sum(len(ld.dataset) for ld in [traindata, validdata, testdata])
    print(f"\nDataset split ({total} total, {args.num_classes}-class)")
    for loader, name in [(traindata, "train"), (validdata, "valid"), (testdata, "test")]:
        split_stats(loader, name)
    print()

    # ========= 2. BUILD MODEL =========
    model = CoMMMultimodalModel(
        mod0_dim=dim0, mod1_dim=dim1,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        seq_len=args.seq_len,
        positional_encoding=args.pos_enc,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ========= 3. TRAIN =========
    model = train_comm(
        model, traindata, validdata,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        save=args.saved_model,
    )

    # ========= 4. TEST =========
    d = test_comm(model, testdata, args.num_classes)
    print_model_metrics(d, args.mod0, args.mod1)

    # ========= 5. EXTRACT REPRESENTATIONS =========
    X_train, y_train = extract_split(model, traindata)
    X_val,   y_val   = extract_split(model, validdata)
    X_test,  y_test  = extract_split(model, testdata)

    # ========= 6. CCS REDUNDANCY =========
    targets = d["true_labels"].long()
    log_py  = compute_log_py(targets, args.num_classes)
    _, ce_list = compute_probs_and_ce(d["pred_modalities"], targets)
    ccs = compute_ccs(ce_list, log_py)
    d["redundancy_ce"]           = ccs.mean().item()
    d["redundancy_pointwise_ce"] = ccs.numpy()

    # ========= 7. SOURCE REDUNDANCY =========
    y_pred_dict = return_redundancy_test_performances(
        X_train, X_val, X_test, y_train.long(), y_val.long(), y_test.long(),
        f"mosei_comm_{pair_name}",
        distribution_target="categorical",
        num_classes=args.num_classes,
    )

    results = compute_redundancy_metrics(y_pred_dict)
    print_model_metrics(d, args.mod0, args.mod1)
    print_redundancy_metrics(results, args.mod0, args.mod1)

    d["source_redundancy_pointwise_ce"] = results["average"]["cross_entropy"]
    d["source_redundancy_preds"]        = y_pred_dict["average"]

    # ========= 8. GLOBAL PID =========
    compute_pid_global(
        d["joint_ce"], d["modalities_ce"][0], d["modalities_ce"][1],
        d["redundancy_ce"], d["source_redundancy_pointwise_ce"],
        num_classes=args.num_classes, targets=d["true_labels"],
        mod0_name=args.mod0, mod1_name=args.mod1,
    )

    # ========= 9. POINTWISE PID =========
    pid_source = compute_pointwise_pid_with_source(d, args.num_classes)
    print(f"\nMean pointwise PID [U_{args.mod0}, U_{args.mod1}, R, S]:")
    print(np.mean(pid_source, axis=0))
    print("Normalised mean:", np.mean(normalize_pid(pid_source), axis=0))

    for i, pid_i in enumerate(pid_source):
        if pid_i[0] < 0 and pid_i[1] >= 0:
            pid_i_copy = [0, pid_i[1], pid_i[2], pid_i[3]+pid_i[0]]
            pid_source[i] = pid_i_copy
        if pid_i[1] < 0 and pid_i[0] >= 0:
            pid_i_copy = [pid_i[0], 0, pid_i[2], pid_i[3]+pid_i[1]]
            pid_source[i] = pid_i_copy

    print(f"\n After correction Mean pointwise PID [U_{args.mod0}, U_{args.mod1}, R, S]:")
    print(np.mean(pid_source, axis=0))
    pid_norm = normalize_pid(pid_source)
    print("After correction Normalised mean:", np.mean(pid_norm, axis=0))