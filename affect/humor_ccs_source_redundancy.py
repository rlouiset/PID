import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pickle

from torch.utils.data import Dataset, DataLoader

from unimodals.common_models import MLP, MLP3, LSTM, Transformer
from synthetic.updated_redundancy_aware_supervised_learning import train, test
from utils_ours import return_redundancy_test_performances
from fusions.common_fusions import Concat

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── Args ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/Users/robinlouiset/Downloads/humor.pkl", type=str)
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--hidden-dim", default=128, type=int)
parser.add_argument("--n-latent", default=256, type=int)
parser.add_argument("--epochs", default=10, type=int) # 40
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--weight-decay", default=1e-2, type=float)
parser.add_argument("--encoder", default="lstm", choices=["lstm", "transformer"],
                    help="Temporal encoder type")
parser.add_argument("--noise-std", default=0.1, type=float,
                    help="Std of Gaussian noise augmentation (0 to disable)")
parser.add_argument("--temporal-dropout-max", default=0.8, type=float,
                    help="Upper bound for dropout rate sampled per sample from U(0, max)")
parser.add_argument("--mod0", default="vision", choices=["vision", "audio", "text"],
                    help="First modality key in the pickle file")
parser.add_argument("--mod1", default="text", choices=["vision", "audio", "text"],
                    help="Second modality key in the pickle file")
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()

assert args.mod0 != args.mod1, "--mod0 and --mod1 must be different"

softmax = torch.nn.Softmax(dim=-1)

# Feature dims per modality key (UR-FUNNY: vision=371, audio=81, text=300)
MODALITY_DIMS = {"vision": 371, "audio": 81, "text": 300}

NUM_CLASSES = 2  # binary: 0=not funny, 1=funny


# ─── Data loading ─────────────────────────────────────────────────────────────

class HumorDataset(Dataset):
    """
    Returns raw temporal sequences (T, D) for two chosen modalities.
    Labels are already binary (0/1). Augmentation: Gaussian noise + temporal dropout.
    """

    def __init__(self, split_dict, mod0_key="vision", mod1_key="text",
                 augment=False, noise_std=0.1, temporal_dropout_max=0.8):
        self.mod0 = torch.from_numpy(split_dict[mod0_key].astype(np.float32))
        self.mod1 = torch.from_numpy(split_dict[mod1_key].astype(np.float32))
        self.labels = torch.from_numpy(split_dict["labels"].squeeze().astype(np.int64))

        self.augment              = augment
        self.noise_std            = noise_std
        self.temporal_dropout_max = temporal_dropout_max

    def _augment(self, x):
        """Gaussian noise + temporal dropout with rate sampled from U(0, max) — CoMM/FactorCL strategy."""
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.temporal_dropout_max > 0:
            rate = float(torch.empty(1).uniform_(0, self.temporal_dropout_max))
            T = x.size(0)
            mask = torch.rand(T) > rate
            x = x * mask.unsqueeze(1).float()
        return x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        m0 = self.mod0[idx]
        m1 = self.mod1[idx]
        if self.augment:
            m0 = self._augment(m0)
            m1 = self._augment(m1)
        return m0, m1, self.labels[idx]


def get_dataloaders(path, mod0_key="vision", mod1_key="text",
                    batch_size=32, num_workers=4,
                    noise_std=0.1, temporal_dropout_max=0.8):
    with open(path, "rb") as f:
        data = pickle.load(f)

    train_ds = HumorDataset(data["train"], mod0_key, mod1_key,
                            augment=True, noise_std=noise_std,
                            temporal_dropout_max=temporal_dropout_max)
    valid_ds = HumorDataset(data["valid"], mod0_key, mod1_key, augment=False)
    test_ds  = HumorDataset(data["test"],  mod0_key, mod1_key, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


# ─── Representation extraction ────────────────────────────────────────────────

def extract_representations(model, dataloader):
    model.eval()
    reps0, reps1, targets = [], [], []
    with torch.no_grad():
        for j in dataloader:
            inputs = [j[0].to(device).float(), j[1].to(device).float()]
            y = j[2]
            _ = model(inputs)
            reps0.append(model.reps[0].cpu())
            reps1.append(model.reps[1].cpu())
            targets.append(y.cpu())
    return {
        "modality0": torch.cat(reps0, dim=0),
        "modality1": torch.cat(reps1, dim=0),
        "targets":   torch.cat(targets, dim=0),
    }


def extract_split(model, loader):
    d = extract_representations(model, loader)
    X = {
        "modality0": d["modality0"].float(),
        "modality1": d["modality1"].float(),
    }
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
    mapping = {
        "modality0": f"Red {mod0_name}",
        "modality1": f"Red {mod1_name}",
        "average":   "Red Joint",
    }
    for key, name in mapping.items():
        acc = results[key]["accuracy"]
        ce  = results[key]["cross_entropy"]
        print(f"{name:<14s} | {acc:8.4f} | {ce:8.4f}")


# ─── PID utilities ────────────────────────────────────────────────────────────

def compute_log_py(targets, num_classes):
    counts = torch.bincount(targets, minlength=num_classes).float()
    probs  = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return torch.log(probs)[targets]


def ce_per_sample(targets, probs, eps=1e-12):
    probs = torch.clamp(probs, eps, 1.0)
    return -torch.log(probs[torch.arange(len(targets)), targets])


def compute_probs_and_ce(logits_list, targets):
    probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
    ce_list    = [ce_per_sample(targets, probs) for probs in probs_list]
    return probs_list, ce_list


def compute_pointwise_information(ce_list, log_py):
    return [-ce - log_py for ce in ce_list]


def compute_ccs_and_selection(ce_list, same_sign, log_py):
    ce_stack = torch.stack(ce_list, dim=1)
    worst_ce = torch.max(ce_stack, dim=1).values
    baseline = -log_py
    return torch.where(same_sign, worst_ce, baseline)


def logp(p):
    return torch.log(torch.clamp(p, 1e-12, 1.0))


def compute_entropy_from_targets(targets, num_classes):
    targets = torch.as_tensor(targets).long()
    counts  = torch.bincount(targets, minlength=num_classes).float()
    probs   = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return -torch.sum(probs * torch.log(probs)).item()


def compute_pid_global(joint_ce, mod0_ce, mod1_ce, red_ce, src_red_ce, targets,
                       mod0_name="mod0", mod1_name="mod1"):
    hy = compute_entropy_from_targets(targets, NUM_CLASSES)

    print(f"H(Y) {hy:.4f}  joint {joint_ce:.4f}  red {red_ce:.4f}  "
          f"src_red {src_red_ce:.4f}  {mod0_name} {mod0_ce:.4f}  {mod1_name} {mod1_ce:.4f}")

    mod0_ce    = min(mod0_ce, hy)
    mod1_ce    = min(mod1_ce, hy)
    red_ce     = min(red_ce,  hy)
    src_red_ce = min(src_red_ce, hy)

    red_ce     = max(red_ce,     joint_ce, mod0_ce, mod1_ce)
    src_red_ce = max(src_red_ce, joint_ce, mod0_ce, mod1_ce)

    red_ce = min(red_ce, src_red_ce)

    mod0_ce = min(max(mod0_ce, joint_ce), red_ce)
    mod1_ce = min(max(mod1_ce, joint_ce), red_ce)

    i_total = hy - joint_ce
    i_r     = hy - red_ce
    i_r_src = hy - src_red_ce
    i_u0    = (hy - mod0_ce) - i_r
    i_u1    = (hy - mod1_ce) - i_r
    i_s     = i_total - i_u0 - i_u1 - i_r

    if i_s < 0:
        i_r     -= i_s
        i_r_src -= i_s
        i_u0 = (hy - mod0_ce) - i_r
        i_u1 = (hy - mod1_ce) - i_r
        i_s  = 0.0

    ratio_src = i_r_src / (i_r + 1e-10)
    print(f"R={i_r:.4f} ({100*ratio_src:.1f}% Source)  "
          f"U_{mod0_name}={i_u0:.4f}  U_{mod1_name}={i_u1:.4f}  S={i_s:.4f}  I={i_total:.4f}")


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
        y = y.long()
        joint_ce = -logp(F.softmax(j,     dim=0))[y]
        mod0_ce  = -logp(F.softmax(m0,    dim=0))[y]
        mod1_ce  = -logp(F.softmax(m1,    dim=0))[y]
        red_ce   = npr
        src_ce   = -logp(F.softmax(r_src, dim=0))[y]
        hy       = -lpy

        joint_ce = min(red_ce, joint_ce)
        src_ce   = min(red_ce, src_ce)
        red_ce   = min(red_ce, hy)
        src_ce   = min(src_ce, hy)
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
    pid = np.maximum(pid, 0)
    pid /= pid.sum(axis=1, keepdims=True) + 1e-12
    return pid


def compute_redundancy_metrics(y_pred_dict):
    results = {}
    for key in ["modality0", "modality1", "average"]:
        acc, ce = traditional_cross_entropy_from_probs(
            softmax(y_pred_dict[key]),
            y_pred_dict["targets"]
        )
        results[key] = {"accuracy": acc, "cross_entropy": ce}
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    dim0 = MODALITY_DIMS[args.mod0]
    dim1 = MODALITY_DIMS[args.mod1]
    pair_name = f"{args.mod0}-{args.mod1}"
    print(f"Loading UR-FUNNY  [{pair_name}]  dims=({dim0}, {dim1})")

    # ========= 1. LOAD DATA =========
    traindata, validdata, testdata = get_dataloaders(
        args.data_path,
        mod0_key=args.mod0,
        mod1_key=args.mod1,
        batch_size=args.bs,
        num_workers=args.num_workers,
        noise_std=args.noise_std,
        temporal_dropout_max=args.temporal_dropout_max,
    )

    # ========= 1b. DATASET STATISTICS =========
    def split_stats(loader, name):
        labels = loader.dataset.labels
        n = len(labels)
        counts = torch.bincount(labels, minlength=NUM_CLASSES)
        probs  = counts.float() / n
        h      = -torch.sum(probs * torch.log(probs.clamp(1e-12))).item()
        majority_acc = counts.max().item() / n
        print(f"  {name:<6s}  n={n:>5d}  "
              f"[not funny: {probs[0]:.3f}  funny: {probs[1]:.3f}]  "
              f"H(Y)={h:.4f}  majority_acc={majority_acc:.4f}")

    total = sum(len(ld.dataset) for ld in [traindata, validdata, testdata])
    print(f"\nDataset split ({total} total)")
    for loader, name in [(traindata, "train"), (validdata, "valid"), (testdata, "test")]:
        split_stats(loader, name)
    print()

    # ========= 2. BUILD MODEL =========
    # Encoders: (batch, 20, D) → (batch, hidden_dim)
    if args.encoder == "lstm":
        encoders = [
            LSTM(dim0, args.hidden_dim).to(device),
            LSTM(dim1, args.hidden_dim).to(device),
        ]
    else:
        encoders = [
            Transformer(dim0, args.hidden_dim).to(device),
            Transformer(dim1, args.hidden_dim).to(device),
        ]

    heads = [
        MLP(args.hidden_dim, args.hidden_dim, NUM_CLASSES).to(device),
        MLP(args.hidden_dim, args.hidden_dim, NUM_CLASSES).to(device),
    ]

    fusion = nn.Sequential(
        Concat(),
        MLP3(2 * args.hidden_dim, args.n_latent, args.n_latent)
    ).to(device)

    head = MLP(args.n_latent, args.hidden_dim, NUM_CLASSES).to(device)

    # ========= 3. TRAIN =========
    model = train(
        encoders,
        fusion,
        head,
        heads,
        traindata,
        validdata,
        args.epochs,
        objective=torch.nn.CrossEntropyLoss(),
        optimtype=torch.optim.AdamW,
        lr=args.lr,
        save=args.saved_model,
        weight_decay=args.weight_decay,
    )

    # ========= 4. TEST =========
    d = test(
        model,
        testdata,
        no_robust=True,
        criterion=torch.nn.CrossEntropyLoss(),
    )
    print_model_metrics(d, args.mod0, args.mod1)

    # ========= 5. EXTRACT REPRESENTATIONS =========
    X_train, y_train = extract_split(model, traindata)
    X_val,   y_val   = extract_split(model, validdata)
    X_test,  y_test  = extract_split(model, testdata)

    # ========= 6. CCS REDUNDANCY =========
    targets     = d["true_labels"].long()
    logits_list = d["pred_modalities"]
    log_py      = compute_log_py(targets, NUM_CLASSES)

    probs_list, ce_list = compute_probs_and_ce(logits_list, targets)
    i_list    = compute_pointwise_information(ce_list, log_py)
    same_sign = torch.sign(i_list[0]) == torch.sign(i_list[1])

    ccs = compute_ccs_and_selection(ce_list, same_sign, log_py)
    d["redundancy_ce"]           = ccs.mean().item()
    d["redundancy_pointwise_ce"] = ccs.numpy()

    # ========= 7. SOURCE REDUNDANCY =========
    y_pred_dict = return_redundancy_test_performances(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        f"humor_ccs_source_{pair_name}",
        distribution_target="categorical",
        num_classes=NUM_CLASSES,
    )

    results = compute_redundancy_metrics(y_pred_dict)
    print_model_metrics(d, args.mod0, args.mod1)
    print_redundancy_metrics(results, args.mod0, args.mod1)

    d["source_redundancy_pointwise_ce"] = results["average"]["cross_entropy"]
    d["source_redundancy_preds"]        = y_pred_dict["average"]

    # ========= 8. GLOBAL PID =========
    compute_pid_global(
        d["joint_ce"],
        d["modalities_ce"][0],
        d["modalities_ce"][1],
        d["redundancy_ce"],
        d["source_redundancy_pointwise_ce"],
        targets=d["true_labels"],
        mod0_name=args.mod0,
        mod1_name=args.mod1,
    )

    # ========= 9. POINTWISE PID =========
    pid_source = compute_pointwise_pid_with_source(d, NUM_CLASSES)

    for i, pid_i in enumerate(pid_source):
        if pid_i[0] < 0 and pid_i[1] >= 0:
            pid_i[-1] += pid_i[0]
            pid_i[0] = 0
        if pid_i[1] < 0 and pid_i[0] >= 0:
            pid_i[-1] += pid_i[1]
            pid_i[1] = 0

            pid_source[i] = pid_i

    print(f"\nMean pointwise PID [U_{args.mod0}, U_{args.mod1}, R, S]:")
    print(np.mean(pid_source, axis=0))
    pid_norm = normalize_pid(pid_source)
    print("Normalised mean:", np.mean(pid_norm, axis=0))
