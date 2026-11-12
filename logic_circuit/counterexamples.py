"""
Reviewer counterexamples (rebuttal: QjeU Q1/Q2, Reviewer 3 Q16).

  --problem twobit   : X1, X2 ~ Bern(0.5) iid,            Y = (X1, X2)        4 classes   (QjeU Q1)
  --problem partial  : X1 = [C, A], X2 = C, C, A iid,     Y = (C, A)          4 classes   (QjeU Q2)
  --problem q16      : X1 = (R0, U1), X2 = (R0, U2),      Y = (R0, U1, U2)    8 classes   (Reviewer 3 Q16)

Original two-bit copy notes:

  X1, X2 ~ independent Bernoulli(0.5),   Y = (X1, X2)  -> 4 equiprobable classes

Expected values (nats):
                         I      I1=I2   R      U1=U2   S      Rsrc   Rmech
  Intuitive / BROJA    1.386   0.693   0      0.693   0      0      0
  Pointwise I_min      1.386   0.693   0.693  0       0.693  0      0.693   <- I_min "same amount" artifact

The question is whether SupGK returns Rsrc ~ 0 (X1 independent of X2 -> no shared information).
We sweep the alignment weight lambda over the values used in the paper.

Estimation follows the OR-gate script (CCS redundancy, same corrections, real SupGK from utils_ours).
Rsrc is reported two ways:
  Rsrc_worst : as in the current code (worst-modality logits from SupGK, no sign filter)
  Rsrc_sign  : with the sign criterion of Eq. 3 applied to the two SupGK heads

Usage:  PYTHONPATH=. python counterexamples.py --problem partial           # full sweep
        PYTHONPATH=. python counterexamples.py --problem partial --quick   # 1 seed, lambda in {0, 10}
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils_ours import return_redundancy_test_performances

LAMBDAS = [0, 1, 10, 25, 50, 100]
LN2 = math.log(2)
PROBLEMS = {
    #            classes  dim1 dim2   expected values (nats) under pointwise I_min + hard GK
    "twobit":  (4, 1, 1, "I=1.386 I1=0.693 I2=0.693 R=0.693 U0=0     U1=0 S=0.693 Rsrc=0     Rmech=0.693"),
    "partial": (4, 2, 1, "I=1.386 I1=1.386 I2=0.693 R=0.693 U0=0.693 U1=0 S=0     Rsrc=0.693 Rmech=0"),
    "q16":     (8, 2, 2, "I=2.079 I1=1.386 I2=1.386 R=1.386 U0=0     U1=0 S=0.693 Rsrc=0.693 Rmech=0.693"),
}
PROBLEM = "twobit"          # set from the command line
NUM_CLASSES = 4


# ----------------------------------------------------------------- data
class CounterexampleDataset(Dataset):
    def __init__(self, n_samples=20000, noise_std=0.1, seed=0):
        g = torch.Generator().manual_seed(seed)
        b = lambda: torch.randint(0, 2, (n_samples, 1), generator=g).float()
        if PROBLEM == "twobit":
            x1, x2 = b(), b()
            self.x1, self.x2 = x1, x2
            y = 2 * x1 + x2
        elif PROBLEM == "partial":
            c, a = b(), b()
            self.x1, self.x2 = torch.cat([c, a], 1), c
            y = 2 * c + a
        elif PROBLEM == "q16":
            r0, u1, u2 = b(), b(), b()
            self.x1, self.x2 = torch.cat([r0, u1], 1), torch.cat([r0, u2], 1)
            y = 4 * r0 + 2 * u1 + u2
        self.y = y.long().squeeze(1)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1, x2 = self.x1[idx], self.x2[idx]
        return (x1 + torch.randn_like(x1) * self.noise_std,
                x2 + torch.randn_like(x2) * self.noise_std,
                self.y[idx])


# ----------------------------------------------------------------- model (LogicNet, 4 classes)
class LogicNet(nn.Module):
    def __init__(self, num_classes, dim1, dim2, repr_dim=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Linear(dim1, 32), nn.ReLU(), nn.Linear(32, repr_dim))
        self.enc2 = nn.Sequential(nn.Linear(dim2, 32), nn.ReLU(), nn.Linear(32, repr_dim))
        self.classifier = nn.Sequential(nn.Linear(2 * repr_dim, 32), nn.ReLU(), nn.Linear(32, num_classes))
        self.head1 = nn.Linear(repr_dim, num_classes)
        self.head2 = nn.Linear(repr_dim, num_classes)

    def forward(self, x1, x2, unimodal=None):
        z1, z2 = self.enc1(x1), self.enc2(x2)
        if unimodal == "modality0":
            return F.log_softmax(self.head1(z1), dim=-1)
        if unimodal == "modality1":
            return F.log_softmax(self.head2(z2), dim=-1)
        joint = self.classifier(torch.cat([z1, z2], dim=-1))
        return (F.log_softmax(joint, dim=-1),
                F.log_softmax(self.head1(z1), dim=-1),
                F.log_softmax(self.head2(z2), dim=-1))

    def get_representations(self, x1, x2):
        return self.enc1(x1), self.enc2(x2)


def train_epoch(model, loader, optimizer, device):
    model.train()
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        joint, m1, m2 = model(x1, x2)
        (F.nll_loss(joint, y) + F.nll_loss(m1, y) + F.nll_loss(m2, y)).backward()
        optimizer.step()


def extract_representations(model, loader, device):
    model.eval()
    z1s, z2s, ys = [], [], []
    with torch.no_grad():
        for x1, x2, y in loader:
            z1, z2 = model.get_representations(x1.to(device), x2.to(device))
            z1s.append(z1.cpu()); z2s.append(z2.cpu()); ys.append(y)
    return torch.cat(z1s), torch.cat(z2s), torch.cat(ys)


def predict_probs(model, loader, device, unimodal=None):
    model.eval()
    out = []
    with torch.no_grad():
        for x1, x2, _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            lp = model(x1, x2)[0] if unimodal is None else model(x1, x2, unimodal)
            out.append(torch.exp(lp).cpu())
    return torch.cat(out)


# ----------------------------------------------------------------- PID (same logic as the OR-gate script)
def logp(p):
    return torch.log(torch.clamp(p, 1e-12, 1.0))


def ce_per_sample(targets, probs):
    return -logp(probs)[torch.arange(len(targets)), targets]


def compute_log_py(targets, num_classes):
    counts = torch.bincount(targets, minlength=num_classes).float()
    return torch.log(torch.clamp(counts / counts.sum(), 1e-12, 1.0))[targets]


def entropy_from_targets(targets, num_classes):
    counts = torch.bincount(targets.long(), minlength=num_classes).float()
    p = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return -torch.sum(p * torch.log(p)).item()


def distribution_pid(joint_ce, m0_ce, m1_ce, redundancy_ce, source_redundancy_ce, num_classes, targets):
    """Same steps as compute_PID_categorical_with_source_decomposition, returning values."""
    H_Y = entropy_from_targets(targets, num_classes)
    I1_raw, I2_raw = H_Y - m0_ce, H_Y - m1_ce
    m0_ce, m1_ce = min(m0_ce, H_Y), min(m1_ce, H_Y)
    redundancy_ce = min(redundancy_ce, H_Y)
    source_redundancy_ce = min(source_redundancy_ce, H_Y)
    redundancy_ce = max(redundancy_ce, joint_ce, m0_ce, m1_ce)
    source_redundancy_ce = max(source_redundancy_ce, joint_ce, m0_ce, m1_ce)
    redundancy_ce = min(redundancy_ce, source_redundancy_ce)
    m0_ce, m1_ce = max(m0_ce, joint_ce), max(m1_ce, joint_ce)
    m0_ce, m1_ce = min(m0_ce, redundancy_ce), min(m1_ce, redundancy_ce)

    I = H_Y - joint_ce
    R = H_Y - redundancy_ce
    Rsrc = H_Y - source_redundancy_ce
    U0, U1 = (H_Y - m0_ce) - R, (H_Y - m1_ce) - R
    S = I - U0 - U1 - R
    if S < 0:
        R -= S; Rsrc -= S
        U0, U1 = (H_Y - m0_ce) - R, (H_Y - m1_ce) - R
        S = 0.0
    return {"I": I, "I1": I1_raw, "I2": I2_raw, "R": R, "Rsrc": Rsrc, "Rmech": R - Rsrc,
            "U0": U0, "U1": U1, "S": S}


# ----------------------------------------------------------------- one run
def run(seed, lam, epochs, device, supgk_epochs_note=""):
    torch.manual_seed(seed); np.random.seed(seed)
    train_set = CounterexampleDataset(20000, seed=seed)
    test_set = CounterexampleDataset(5000, seed=seed + 10_000)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    train_eval = DataLoader(train_set, batch_size=512)
    test_loader = DataLoader(test_set, batch_size=512)

    _, d1, d2, _ = PROBLEMS[PROBLEM]
    model = LogicNet(NUM_CLASSES, d1, d2).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_epoch(model, train_loader, opt, device)

    z1_tr, z2_tr, y_tr = extract_representations(model, train_eval, device)
    z1_te, z2_te, y_te = extract_representations(model, test_loader, device)
    pj = predict_probs(model, test_loader, device)
    p0 = predict_probs(model, test_loader, device, "modality0")
    p1 = predict_probs(model, test_loader, device, "modality1")

    # CCS total redundancy (Eq. 3)
    log_py = compute_log_py(y_te, NUM_CLASSES)
    ce_list = [ce_per_sample(y_te, p0), ce_per_sample(y_te, p1)]
    i0, i1 = [-ce - log_py for ce in ce_list]
    same_sign = torch.sign(i0) == torch.sign(i1)
    worst_ce = torch.max(torch.stack(ce_list, 1), 1).values
    ccs = torch.where(same_sign, worst_ce, -log_py)

    # SupGK (real implementation)
    y_pred = return_redundancy_test_performances(
        {"modality0": z1_tr.float(), "modality1": z2_tr.float()},
        {"modality0": z1_te.float(), "modality1": z2_te.float()},
        {"modality0": z1_te.float(), "modality1": z2_te.float()},
        y_tr, y_te, y_te, f"{PROBLEM}_seed{seed}_lam{lam}",
        distribution_target="categorical", num_classes=NUM_CLASSES, h_dim=1024, lambda_reg=lam,
    )
    yt = y_pred["targets"]
    lpy_t = compute_log_py(yt, NUM_CLASSES)

    # Rsrc as in the current code: worst-modality logits
    src_ce_worst = ce_per_sample(yt, F.softmax(y_pred["average"], -1)).mean().item()
    # Rsrc with sign criterion on the two SupGK heads
    j0 = -ce_per_sample(yt, F.softmax(y_pred["modality0"], -1)) - lpy_t
    j1 = -ce_per_sample(yt, F.softmax(y_pred["modality1"], -1)) - lpy_t
    rsrc_sign = torch.where(j0 * j1 > 0, torch.minimum(j0, j1), torch.zeros_like(j0)).mean().item()

    joint_ce = ce_per_sample(y_te, pj).mean().item()
    d = distribution_pid(joint_ce, ce_list[0].mean().item(), ce_list[1].mean().item(),
                         ccs.mean().item(), src_ce_worst, NUM_CLASSES, y_te)
    d["R_uncorrected"] = (-log_py.mean() - ccs.mean()).item()
    d["Rsrc_sign"] = rsrc_sign
    d["supgk_I0"], d["supgk_I1"] = j0.mean().item(), j1.mean().item()
    return d


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--problem", choices=list(PROBLEMS), default="twobit")
    args = ap.parse_args()
    global PROBLEM, NUM_CLASSES
    PROBLEM = args.problem
    NUM_CLASSES = PROBLEMS[PROBLEM][0]
    lambdas = LAMBDAS
    if args.quick:
        args.seeds, args.epochs, lambdas = 3, 100, [0, 1, 10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cols = ["I", "R_uncorrected", "R", "Rsrc", "Rsrc_sign", "Rmech", "U0", "U1", "S", "supgk_I0", "supgk_I1"]
    results = {}
    for lam in lambdas:
        runs = [run(s, lam, args.epochs, device) for s in range(args.seeds)]
        results[lam] = {c: (np.mean([r[c] for r in runs]), np.std([r[c] for r in runs])) for c in cols}

    print(f"\n{'=' * 60}\nProblem: {PROBLEM} — all values in nats (mean ± std over seeds)")
    print("Expected: " + PROBLEMS[PROBLEM][3] + "\n")
    print(f"{'lambda':>6} | " + " ".join(f"{c:>15}" for c in cols))
    for lam in lambdas:
        print(f"{lam:>6} | " + " ".join(f"{m:7.3f}±{s:6.3f}" for m, s in (results[lam][c] for c in cols)))
    print("\nsupgk_I0 / supgk_I1: information about Y carried by each SupGK representation (should not exceed Rsrc*).")
    np.save(f"{PROBLEM}_results.npy", results, allow_pickle=True)


if __name__ == "__main__":
    main()