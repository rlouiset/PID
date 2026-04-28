"""
LSMI-based pointwise PID estimation for UR-FUNNY with CoMM architecture.
  1. Train CoMM model (Conv1d + Transformer encoder + fusion transformer)
  2. Extract mean-pooled representations  (embed_dim-d per modality)
  3. Estimate pointwise PID via LSMI:
       - Discriminators  →  I(Xi ; Y)
       - MargKernel      →  H(Xi)
       - R = min(H1,H2) – min(H1–I1Y, H2–I2Y)
Reference: Dufumier et al. 2025 (CoMM), av_mnist/avmnist_lsmi.py
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pickle

from torch.utils.data import Dataset, DataLoader

from utils_lsmi import MargKernel, cls_network, feature_dataset, setup_seed

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── Args ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
# Data / CoMM
parser.add_argument("--data-path", default="/home/rlouiset/humor.pkl", type=str)
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--embed-dim", default=40, type=int,
                    help="CoMM Transformer embedding dim")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--weight-decay", default=1e-2, type=float)
parser.add_argument("--seq-len", default=30, type=int,
                    help="Sequence length for UR-FUNNY")
parser.add_argument("--pos-enc", action="store_true")
parser.add_argument("--noise-std", default=0.1, type=float)
parser.add_argument("--temporal-dropout-max", default=0.8, type=float)
parser.add_argument("--mod0", default="vision", choices=["vision", "audio", "text"])
parser.add_argument("--mod1", default="text",   choices=["vision", "audio", "text"])
parser.add_argument("--saved-model", default=None, type=str)
# LSMI
parser.add_argument("--lsmi-embed-size", default=128, type=int,
                    help="Hidden dim for LSMI discriminators and joint classifier")
parser.add_argument("--lsmi-bs", default=512, type=int,
                    help="Batch size for LSMI estimator training / inference")
parser.add_argument("--epochs-discriminator", default=30, type=int)
parser.add_argument("--epochs-entropy", default=30, type=int)
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()

assert args.mod0 != args.mod1, "--mod0 and --mod1 must be different"

# UR-FUNNY feature dims (same extraction pipeline as MUsTARD)
MODALITY_DIMS = {"vision": 371, "audio": 81, "text": 300}
NUM_CLASSES   = 2  # binary: 0=not funny, 1=funny


# ─── Data loading ─────────────────────────────────────────────────────────────

class HumorDataset(Dataset):
    """Labels are already binary (0/1). Augmentation: Gaussian noise + temporal dropout."""
    def __init__(self, split_dict, mod0_key="vision", mod1_key="text",
                 augment=False, noise_std=0.1, temporal_dropout_max=0.8):
        self.mod0   = torch.from_numpy(split_dict[mod0_key].astype(np.float32))
        self.mod1   = torch.from_numpy(split_dict[mod1_key].astype(np.float32))
        self.labels = torch.from_numpy(split_dict["labels"].squeeze().astype(np.int64))
        self.augment              = augment
        self.noise_std            = noise_std
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


def get_dataloaders(path, mod0_key, mod1_key,
                    batch_size=32, num_workers=4, noise_std=0.1, temporal_dropout_max=0.8):
    with open(path, "rb") as f:
        data = pickle.load(f)
    train_ds       = HumorDataset(data["train"], mod0_key, mod1_key,
                                  augment=True, noise_std=noise_std,
                                  temporal_dropout_max=temporal_dropout_max)
    valid_ds       = HumorDataset(data["valid"], mod0_key, mod1_key, augment=False)
    test_ds        = HumorDataset(data["test"],  mod0_key, mod1_key, augment=False)
    train_clean_ds = HumorDataset(data["train"], mod0_key, mod1_key, augment=False)
    return (
        DataLoader(train_ds,       batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(valid_ds,       batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,        batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(train_clean_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# ─── CoMM Architecture ───────────────────────────────────────────────────────

def build_sincos_posemb(max_seq_len, embed_dim):
    assert embed_dim % 2 == 0
    pe  = torch.zeros(max_seq_len, embed_dim)
    pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class CoMMEncoder(nn.Module):
    def __init__(self, n_features, embed_dim=40, max_seq_len=30,
                 n_heads=5, n_layers=5, positional_encoding=False):
        super().__init__()
        self.conv = nn.Conv1d(n_features, embed_dim, kernel_size=1, bias=False)
        self.use_pos_enc = positional_encoding
        if positional_encoding:
            self.register_buffer("pos_emb", build_sincos_posemb(max_seq_len, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_pos_enc:
            x = x + self.pos_emb[:, :x.size(1)]
        return self.transformer(x)


class FusionTransformer(nn.Module):
    def __init__(self, embed_dim=40, n_heads=8, n_layers=1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, sequences):
        batch  = sequences[0].size(0)
        cls    = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls] + sequences, dim=1)
        return self.norm(self.transformer(tokens)[:, 0])


class CoMMMultimodalModel(nn.Module):
    def __init__(self, mod0_dim, mod1_dim, num_classes,
                 embed_dim=40, seq_len=30, positional_encoding=False):
        super().__init__()
        self.encoders = nn.ModuleList([
            CoMMEncoder(mod0_dim, embed_dim, seq_len, positional_encoding=positional_encoding),
            CoMMEncoder(mod1_dim, embed_dim, seq_len, positional_encoding=positional_encoding),
        ])
        self.fusion    = FusionTransformer(embed_dim)
        self.head      = nn.Linear(embed_dim, num_classes)
        self.mod_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes),
            nn.Linear(embed_dim, num_classes),
        ])
        self.reps = []

    def forward(self, inputs):
        seqs         = [enc(x.float()) for enc, x in zip(self.encoders, inputs)]
        self.reps    = seqs
        fused        = self.fusion(seqs)
        joint_logits = self.head(fused)
        mod_logits   = [h(s.mean(dim=1)) for h, s in zip(self.mod_heads, seqs)]
        return joint_logits, seqs, mod_logits


# ─── Training ─────────────────────────────────────────────────────────────────

def train_comm(model, traindata, validdata, epochs, lr, weight_decay, save=None):
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_model = copy.deepcopy(model)
    best_val   = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; n = 0
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
        val_loss = 0.0; val_n = 0; val_correct = 0
        with torch.no_grad():
            for m0, m1, y in validdata:
                inputs = [m0.to(device), m1.to(device)]
                y = y.to(device).long()
                joint, _, _ = model(inputs)
                val_loss    += criterion(joint, y).item() * len(y)
                val_n       += len(y)
                val_correct += (joint.argmax(1) == y).sum().item()
        val_loss /= val_n
        print(f"Epoch {epoch}  train={total_loss/n:.4f}  val={val_loss:.4f}  acc={val_correct/val_n:.4f}")

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
    pred_joint, pred_m0, pred_m1, true_labels = [], [], [], []
    total_joint = 0.0; total_m0 = 0.0; total_m1 = 0.0; n = 0

    with torch.no_grad():
        for m0, m1, y in testdata:
            inputs = [m0.to(device), m1.to(device)]
            y = y.to(device).long()
            joint, _, mod_logits = model(inputs)
            total_joint += criterion(joint,         y).item() * len(y)
            total_m0    += criterion(mod_logits[0], y).item() * len(y)
            total_m1    += criterion(mod_logits[1], y).item() * len(y)
            n           += len(y)
            pred_joint.append(joint.cpu()); pred_m0.append(mod_logits[0].cpu())
            pred_m1.append(mod_logits[1].cpu()); true_labels.append(y.cpu())

    pred_joint  = torch.cat(pred_joint)
    true_labels = torch.cat(true_labels)

    def acc(t, p): return (p.argmax(1) == t).float().mean().item()
    print(f"{'Model':<12s} | {'Acc':>8s} | {'CE':>8s}")
    print("-" * 36)
    print(f"{'Joint':<12s} | {acc(true_labels, pred_joint):8.4f} | {total_joint/n:8.4f}")
    print(f"{args.mod0:<12s} | {acc(true_labels, torch.cat(pred_m0)):8.4f} | {total_m0/n:8.4f}")
    print(f"{args.mod1:<12s} | {acc(true_labels, torch.cat(pred_m1)):8.4f} | {total_m1/n:8.4f}")


# ─── Representation extraction ────────────────────────────────────────────────

def extract_representations(model, loader):
    """Extract mean-pooled CoMM sequences → (N, embed_dim) per modality."""
    model.eval()
    reps0, reps1, targets = [], [], []
    with torch.no_grad():
        for m0, m1, y in loader:
            _ = model([m0.to(device), m1.to(device)])
            reps0.append(model.reps[0].mean(dim=1).cpu())
            reps1.append(model.reps[1].mean(dim=1).cpu())
            targets.append(y.cpu())
    return (torch.cat(reps0).float(),
            torch.cat(reps1).float(),
            torch.cat(targets).long())


# ─── LSMI estimators ─────────────────────────────────────────────────────────

def obtain_discriminator(train_loader, input_size_1, input_size_2, embed_size, n_classes, n_epochs):
    """Train three classifiers: mod0→Y, mod1→Y, (mod0,mod1)→Y."""
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
    """Train two MargKernel models to estimate H(mod0) and H(mod1)."""
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
            info.append(np.log(n_classes) + F.log_softmax(out, dim=1)[rows, y])
    return torch.cat(info).detach()


def get_entropy(loader, model, modality):
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            x = (batch[0] if modality == 'modality_1' else batch[1]).to(device)
            info.append(model(x))
    return torch.cat(info).detach()


def LSMI_estimation(loader, discriminators, entropy_estimators, split_name=""):
    """Compute pointwise PID via LSMI and apply RUS adjustment."""
    I1Y  = get_mutual_info(loader, discriminators[0], 'modality_1',  NUM_CLASSES)
    I2Y  = get_mutual_info(loader, discriminators[1], 'modality_2',  NUM_CLASSES)
    I12Y = get_mutual_info(loader, discriminators[2], 'modality_12', NUM_CLASSES)
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


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_seed(args.seed)

    dim0 = MODALITY_DIMS[args.mod0]
    dim1 = MODALITY_DIMS[args.mod1]
    pair_name = f"{args.mod0}-{args.mod1}"
    print(f"Loading UR-FUNNY  [{pair_name}]  dims=({dim0},{dim1})  embed={args.embed_dim}")

    # ========= 1. LOAD DATA =========
    traindata, validdata, testdata, train_clean = get_dataloaders(
        args.data_path, args.mod0, args.mod1,
        batch_size=args.bs, num_workers=args.num_workers,
        noise_std=args.noise_std, temporal_dropout_max=args.temporal_dropout_max,
    )

    def split_stats(loader, name):
        labels = loader.dataset.labels
        n = len(labels)
        counts = torch.bincount(labels, minlength=NUM_CLASSES)
        probs  = counts.float() / n
        h      = -torch.sum(probs * torch.log(probs.clamp(1e-12))).item()
        print(f"  {name:<6s}  n={n:>5d}  "
              f"[not funny: {probs[0]:.3f}  funny: {probs[1]:.3f}]  "
              f"H(Y)={h:.4f}  majority_acc={counts.max().item()/n:.4f}")

    total = sum(len(ld.dataset) for ld in [traindata, validdata, testdata])
    print(f"\nDataset split ({total} total)")
    for loader, name in [(traindata, "train"), (validdata, "valid"), (testdata, "test")]:
        split_stats(loader, name)
    print()

    # ========= 2. BUILD + TRAIN CoMM =========
    model = CoMMMultimodalModel(
        mod0_dim=dim0, mod1_dim=dim1,
        num_classes=NUM_CLASSES,
        embed_dim=args.embed_dim,
        seq_len=args.seq_len,
        positional_encoding=args.pos_enc,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model = train_comm(model, traindata, validdata,
                       epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                       save=args.saved_model)

    # ========= 3. TEST CoMM =========
    print("\nTest performance:")
    test_comm(model, testdata)

    # ========= 4. EXTRACT REPRESENTATIONS =========
    # Use clean (non-augmented) train loader so LSMI sees the true data distribution
    print("\nExtracting representations...")
    train_r1, train_r2, y_train = extract_representations(model, train_clean)
    val_r1,   val_r2,   y_val   = extract_representations(model, validdata)
    test_r1,  test_r2,  y_test  = extract_representations(model, testdata)
    print(f"  Representation dim: {train_r1.shape[1]}")

    # ========= 5. BUILD LSMI FEATURE LOADERS =========
    lsmi_train = DataLoader(
        feature_dataset(train_r1, train_r2, y_train),
        batch_size=args.lsmi_bs, shuffle=True,  num_workers=0)
    lsmi_val   = DataLoader(
        feature_dataset(val_r1,   val_r2,   y_val),
        batch_size=args.lsmi_bs, shuffle=False, num_workers=0)
    lsmi_test  = DataLoader(
        feature_dataset(test_r1,  test_r2,  y_test),
        batch_size=args.lsmi_bs, shuffle=False, num_workers=0)

    feat_dim_1 = train_r1.shape[1]  # == args.embed_dim
    feat_dim_2 = train_r2.shape[1]

    # ========= 6. TRAIN LSMI ESTIMATORS =========
    print("\nTraining LSMI discriminators...")
    discriminators = obtain_discriminator(
        lsmi_train, feat_dim_1, feat_dim_2,
        embed_size=args.lsmi_embed_size,
        n_classes=NUM_CLASSES,
        n_epochs=args.epochs_discriminator,
    )

    print("\nTraining LSMI entropy estimators...")
    entropy_estimators = obtain_entropy_estimator(
        lsmi_train, feat_dim_1, feat_dim_2,
        n_epochs=args.epochs_entropy,
    )

    # ========= 7. LSMI PID ESTIMATION =========
    print("\nDistribution-Level PID:")
    LSMI_estimation(lsmi_train, discriminators, entropy_estimators, "train")
    LSMI_estimation(lsmi_val,   discriminators, entropy_estimators, "val")
    r, u1, u2, s = LSMI_estimation(lsmi_test, discriminators, entropy_estimators, "test")

    # ========= 8. POINTWISE DISTRIBUTION =========
    pid = np.stack([u1.cpu().numpy(), u2.cpu().numpy(),
                    r.cpu().numpy(),  s.cpu().numpy()], axis=1)

    pid_clipped = np.maximum(pid, 0)
    pid_norm = pid_clipped / (pid_clipped.sum(axis=1, keepdims=True) + 1e-12)

    print(f"\nMean pointwise PID [U_{args.mod0}, U_{args.mod1}, R, S] (test):")
    print(np.mean(pid, axis=0))
    print("Normalised mean:", np.mean(pid_norm, axis=0))

    """for i, pid_i in enumerate(pid):
        if pid_i[0] < 0 and pid_i[1] >= 0:
            pid_i[-1] += pid_i[0]; pid_i[0] = 0
        if pid_i[1] < 0 and pid_i[0] >= 0:
            pid_i[-1] += pid_i[1]; pid_i[1] = 0
        pid[i] = pid_i"""

    for i, pid_i in enumerate(pid):
        if pid_i[0] < 0 and pid_i[1] >= 0:
            pid_i_copy = [0, pid_i[1], pid_i[2], pid_i[3] + pid_i[0]]
            pid[i] = pid_i_copy
        if pid_i[1] < 0 and pid_i[0] >= 0:
            pid_i_copy = [pid_i[0], 0, pid_i[2], pid_i[3] + pid_i[1]]
            pid[i] = pid_i_copy

    pid_clipped = np.maximum(pid, 0)
    pid_norm = pid_clipped / (pid_clipped.sum(axis=1, keepdims=True) + 1e-12)
    print(f"\nAfter correction  Mean pointwise PID [U_{args.mod0}, U_{args.mod1}, R, S]:")
    print(np.mean(pid, axis=0))
    print("After correction  Normalised mean:", np.mean(pid_norm, axis=0))
