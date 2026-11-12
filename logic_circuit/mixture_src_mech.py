"""
Controlled mixture of source and mechanistic redundancy (rebuttal: L8Vf W2/W3, QjeU Q3).

Latent factors
  C            : shared bit, P(C=1) = p_c          -> present in X1 and X2, copied into Y  (source redundancy)
  A1, A2       : m independent private bits each   -> combined through m OR gates into Y   (mechanistic redundancy)
Observations (+ Gaussian noise, as in ORDataset)
  X1 = [C, A1_1..A1_m],   X2 = [C, A2_1..A2_m]
Target
  Y  = (C, OR(A1_1,A2_1), ..., OR(A1_m,A2_m))  encoded as a single class index

Ground truth (exact enumeration over the latent distribution, in nats as in the estimator):
  R_true     = E[ min(i1, i2) if i1*i2 > 0 else 0 ]      (pointwise I_min with sign criterion, Eq. 3)
  Rsrc_true  = I(C;Y) = H(C)                              (hard Gacs-Korner: Z^C = C)
  Rmech_true = R_true - Rsrc_true

Estimation follows the OR-gate script exactly (CCS redundancy, same clipping/corrections,
SupGK via utils_ours.return_redundancy_test_performances).

Usage:  python mixture_source_mechanistic.py            # full sweep
        python mixture_source_mechanistic.py --quick    # 1 seed, 20 epochs (smoke test)
"""

import argparse
import itertools
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import logging
import warnings

warnings.filterwarnings("ignore")
for _name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from utils_ours import return_redundancy_test_performances

softmax = torch.nn.Softmax(dim=-1)

# Settings: (p_c, m).  p_c=None -> no shared factor (pure mechanistic); m=0 -> no OR gate (pure source)
N_UNIQUE = 0   # number of unique bits seen only by modality 1 (set with --unique)
LAMBDA = 10    # SupGK alignment weight (set with --lam; 10 = utils_ours default)

SETTINGS = [
    (None, 1),   # 0% source
    (0.05, 3),   # ~31%
    (0.05, 1),   # ~49%
    (0.10, 2),   # ~56%
    (0.20, 2),   # ~69%
    (0.50, 2),   # ~79%
    (0.50, 1),   # ~94%
    (0.50, 0),   # 100% source
]


# =====================================================================
# Data
# =====================================================================
def encode_y(c, or_bits, has_c):
    idx = 0
    for j, b in enumerate(or_bits):
        idx += int(b) << j
    if has_c:
        idx += int(c) << len(or_bits)
    return idx


class MixtureDataset(Dataset):
    def __init__(self, p_c, m, n_samples=20000, noise_std=0.1, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.has_c = p_c is not None
        self.m = m
        n = n_samples

        c = (torch.rand(n, 1, generator=g) < p_c).float() if self.has_c else torch.zeros(n, 0)
        a1 = torch.randint(0, 2, (n, m), generator=g).float()
        a2 = torch.randint(0, 2, (n, m), generator=g).float()
        or_bits = ((a1 + a2) > 0).long()
        u = torch.randint(0, 2, (n, N_UNIQUE), generator=g).float()   # unique bits, modality 1 only

        self.x1 = torch.cat([c, a1, u], dim=1)
        self.x2 = torch.cat([c, a2], dim=1)
        c_col = c[:, 0].long() if self.has_c else torch.zeros(n, dtype=torch.long)
        base = (2 if self.has_c else 1) * (2 ** m)
        u_idx = (u.long() * (2 ** torch.arange(N_UNIQUE))).sum(1) if N_UNIQUE else torch.zeros(n, dtype=torch.long)
        self.y = torch.tensor(
            [encode_y(c_col[i], or_bits[i].tolist(), self.has_c) for i in range(n)], dtype=torch.long
        ) + base * u_idx
        self.noise_std = noise_std
        self.num_classes = base * (2 ** N_UNIQUE)
        self.in_dim1, self.in_dim2 = self.x1.shape[1], self.x2.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1, x2 = self.x1[idx], self.x2[idx]
        return (x1 + torch.randn_like(x1) * self.noise_std,
                x2 + torch.randn_like(x2) * self.noise_std,
                self.y[idx])


# =====================================================================
# Exact ground truth (nats)
# =====================================================================
def ground_truth(p_c, m):
    has_c = p_c is not None
    P = defaultdict(float)
    c_vals = [0, 1] if has_c else [0]
    for c in c_vals:
        pc = (p_c if c else 1 - p_c) if has_c else 1.0
        for a1 in itertools.product([0, 1], repeat=m):
            for a2 in itertools.product([0, 1], repeat=m):
                for uu in itertools.product([0, 1], repeat=N_UNIQUE):
                    ors = tuple(p | q for p, q in zip(a1, a2))
                    P[((c, a1, uu), (c, a2), (c, ors, uu))] += pc / (4 ** m) / (2 ** N_UNIQUE)

    def marg(f):
        d = defaultdict(float)
        for s, p in P.items():
            d[f(s)] += p
        return d

    py, p1, p2 = marg(lambda s: s[2]), marg(lambda s: s[0]), marg(lambda s: s[1])
    p1y, p2y = marg(lambda s: (s[0], s[2])), marg(lambda s: (s[1], s[2]))

    R = I1 = I2 = 0.0
    for (x1, x2, y), p in P.items():
        i1 = math.log(p1y[(x1, y)] / p1[x1] / py[y])
        i2 = math.log(p2y[(x2, y)] / p2[x2] / py[y])
        R += p * (min(i1, i2) if i1 * i2 > 0 else 0.0)
        I1 += p * i1
        I2 += p * i2
    Rsrc = -(p_c * math.log(p_c) + (1 - p_c) * math.log(1 - p_c)) if has_c else 0.0
    return {"I1": I1, "I2": I2, "R": R, "Rsrc": Rsrc, "Rmech": R - Rsrc, "U0": I1 - R, "U1": I2 - R}


# =====================================================================
# Model / training (generalized LogicNet)
# =====================================================================
class LogicNet(nn.Module):
    def __init__(self, in_dim1, in_dim2, num_classes, repr_dim=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Linear(in_dim1, 32), nn.ReLU(), nn.Linear(32, repr_dim))
        self.enc2 = nn.Sequential(nn.Linear(in_dim2, 32), nn.ReLU(), nn.Linear(32, repr_dim))
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
        loss = F.nll_loss(joint, y) + F.nll_loss(m1, y) + F.nll_loss(m2, y)
        loss.backward()
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
            logp = model(x1, x2)[0] if unimodal is None else model(x1, x2, unimodal)
            out.append(torch.exp(logp).cpu())
    return torch.cat(out)


# =====================================================================
# PID estimation (identical logic to the OR-gate script)
# =====================================================================
def logp(p):
    return torch.log(torch.clamp(p, 1e-12, 1.0))


def ce_mean(probs, targets):
    return -logp(probs)[torch.arange(len(targets)), targets].mean().item()


def ce_per_sample(targets, probs):
    return -logp(probs)[torch.arange(len(targets)), targets]


def compute_log_py(targets, num_classes):
    counts = torch.bincount(targets, minlength=num_classes).float()
    probs = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return torch.log(probs)[targets]


def entropy_from_targets(targets, num_classes):
    counts = torch.bincount(targets.long(), minlength=num_classes).float()
    probs = torch.clamp(counts / counts.sum(), 1e-12, 1.0)
    return -torch.sum(probs * torch.log(probs)).item()


def distribution_pid(joint_ce, m0_ce, m1_ce, redundancy_ce, source_redundancy_ce, num_classes, targets,
                     mode="max"):
    """
    mode="max": current method (as in compute_PID_categorical_with_source_decomposition):
                R <- max(R, Rsrc); negative synergy shifted into both R and Rsrc.
    mode="min": alternative: R left untouched, Rsrc <- min(Rsrc, R);
                negative synergy shifted into R only, then Rsrc re-capped at R.
    """
    H_Y = entropy_from_targets(targets, num_classes)
    m0_ce, m1_ce = min(m0_ce, H_Y), min(m1_ce, H_Y)
    redundancy_ce = min(redundancy_ce, H_Y)
    source_redundancy_ce = min(source_redundancy_ce, H_Y)

    redundancy_ce = max(redundancy_ce, joint_ce, m0_ce, m1_ce)
    source_redundancy_ce = max(source_redundancy_ce, joint_ce, m0_ce, m1_ce)
    activated = source_redundancy_ce < redundancy_ce          # i.e. Rsrc > R before correction
    if mode == "max":
        redundancy_ce = min(redundancy_ce, source_redundancy_ce)   # R <- max(R, Rsrc)
    else:
        source_redundancy_ce = max(source_redundancy_ce, redundancy_ce)   # Rsrc <- min(Rsrc, R)

    m0_ce, m1_ce = max(m0_ce, joint_ce), max(m1_ce, joint_ce)
    m0_ce, m1_ce = min(m0_ce, redundancy_ce), min(m1_ce, redundancy_ce)

    I = H_Y - joint_ce
    I_R = H_Y - redundancy_ce
    I_Rsrc = H_Y - source_redundancy_ce
    I_U0 = (H_Y - m0_ce) - I_R
    I_U1 = (H_Y - m1_ce) - I_R
    I_S = I - I_U0 - I_U1 - I_R
    if I_S < 0:
        I_R -= I_S
        if mode == "max":
            I_Rsrc -= I_S
        I_U0 = (H_Y - m0_ce) - I_R
        I_U1 = (H_Y - m1_ce) - I_R
        I_S = 0.0
    if mode == "min":
        I_Rsrc = min(I_Rsrc, I_R)
    return {"H_Y": H_Y, "I": I, "R": I_R, "Rsrc": I_Rsrc, "Rmech": I_R - I_Rsrc,
            "U0": I_U0, "U1": I_U1, "S": I_S, "activated": float(activated)}


def pointwise_quantities(probs_joint, probs_m0, probs_m1, ccs_ce, y_pred, targets, num_classes):
    """
    Per-sample quantities (nats):
      r        : total redundancy from Eq. 3 (CCS), uncorrected
      rsrc     : source redundancy from SupGK worst-modality logits (current code, no sign filter)
      rsrc_sgn : source redundancy with the sign criterion applied to the two SupGK heads
      i0, i1, i_joint : pointwise information of each unimodal / joint predictor
    """
    log_py = compute_log_py(targets, num_classes)
    h_y = -log_py
    r = (h_y - torch.as_tensor(ccs_ce)).numpy()
    rsrc = (h_y - ce_per_sample(targets, softmax(y_pred["average"]))).numpy()
    j0 = h_y - ce_per_sample(targets, softmax(y_pred["modality0"]))
    j1 = h_y - ce_per_sample(targets, softmax(y_pred["modality1"]))
    rsrc_sgn = torch.where(j0 * j1 > 0, torch.minimum(j0, j1), torch.zeros_like(j0)).numpy()
    i0 = (h_y - ce_per_sample(targets, probs_m0)).numpy()
    i1 = (h_y - ce_per_sample(targets, probs_m1)).numpy()
    return {"r": r, "rsrc": rsrc, "rsrc_sgn": rsrc_sgn, "i0": i0, "i1": i1}


def pointwise_variants(q):
    """Means of pointwise R, Rsrc, U0, U1 under the two corrections."""
    r, rsrc, rsrc_sgn, i0, i1 = q["r"], q["rsrc"], q["rsrc_sgn"], q["i0"], q["i1"]
    out = {"activation": 100.0 * np.mean(rsrc > r)}
    # current: r <- max(r, rsrc), rsrc unchanged
    r_max = np.maximum(r, rsrc)
    out.update({"R_max": r_max.mean(), "Rsrc_max": rsrc.mean(),
                "U0_max": (i0 - r_max).mean(), "U1_max": (i1 - r_max).mean(),
                "negU_max": 100.0 * np.mean((i0 - r_max < -1e-6) | (i1 - r_max < -1e-6))})
    # alternative: r unchanged, rsrc <- min(rsrc, r)
    rsrc_min = np.minimum(rsrc, r)
    out.update({"R_min": r.mean(), "Rsrc_min": rsrc_min.mean(),
                "U0_min": (i0 - r).mean(), "U1_min": (i1 - r).mean(),
                "negU_min": 100.0 * np.mean((i0 - r < -1e-6) | (i1 - r < -1e-6))})
    # alternative + sign criterion on SupGK heads
    rsrc_min_sgn = np.minimum(rsrc_sgn, r)
    out.update({"Rsrc_minsgn": rsrc_min_sgn.mean()})
    return out


# =====================================================================
# One run
# =====================================================================
def run_setting(p_c, m, seed, epochs, device):
    torch.manual_seed(seed); np.random.seed(seed)
    train_set = MixtureDataset(p_c, m, 20000, seed=seed)
    test_set = MixtureDataset(p_c, m, 5000, seed=seed + 10_000)
    K = train_set.num_classes

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    train_loader_eval = DataLoader(train_set, batch_size=512)
    test_loader = DataLoader(test_set, batch_size=512)

    model = LogicNet(train_set.in_dim1, train_set.in_dim2, K).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        train_epoch(model, train_loader, opt, device)

    z1_tr, z2_tr, y_tr = extract_representations(model, train_loader_eval, device)
    z1_te, z2_te, y_te = extract_representations(model, test_loader, device)

    pj = predict_probs(model, test_loader, device)
    p0 = predict_probs(model, test_loader, device, "modality0")
    p1 = predict_probs(model, test_loader, device, "modality1")

    # CCS total redundancy (Eq. 3)
    log_py = compute_log_py(y_te, K)
    ce_list = [ce_per_sample(y_te, p0), ce_per_sample(y_te, p1)]
    i_list = [-ce - log_py for ce in ce_list]
    same_sign = torch.sign(i_list[0]) == torch.sign(i_list[1])
    worst_ce = torch.max(torch.stack(ce_list, dim=1), dim=1).values
    ccs = torch.where(same_sign, worst_ce, -log_py)

    # SupGK source redundancy
    y_pred = return_redundancy_test_performances(
        {"modality0": z1_tr.float(), "modality1": z2_tr.float()},
        {"modality0": z1_te.float(), "modality1": z2_te.float()},
        {"modality0": z1_te.float(), "modality1": z2_te.float()},
        y_tr, y_te, y_te, f"mixture_pc{p_c}_m{m}_u{N_UNIQUE}_lam{LAMBDA}_seed{seed}",
        distribution_target="categorical", num_classes=K, h_dim=1024, lambda_reg=LAMBDA,
    )
    assert torch.equal(y_pred["targets"].cpu(), y_te.cpu()), "SupGK test order differs from y_te"
    src_ce = ce_mean(softmax(y_pred["average"]), y_pred["targets"])

    args = (ce_mean(pj, y_te), ce_mean(p0, y_te), ce_mean(p1, y_te), ccs.mean().item(), src_ce, K, y_te)
    dist_max = distribution_pid(*args, mode="max")
    dist_min = distribution_pid(*args, mode="min")
    pw = pointwise_variants(pointwise_quantities(pj, p0, p1, ccs, y_pred, y_te, K))
    return dist_max, dist_min, pw


# =====================================================================
# Main
# =====================================================================
def frac(src, r):
    return 100.0 * src / r if abs(r) > 1e-8 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--unique", type=int, default=0,
                    help="number of unique bits seen only by modality 1 and copied into Y (QjeU Q3)")
    ap.add_argument("--lam", type=float, default=10, help="SupGK alignment weight lambda")
    args = ap.parse_args()
    global N_UNIQUE, LAMBDA
    N_UNIQUE, LAMBDA = args.unique, args.lam
    if args.quick:
        args.seeds, args.epochs = 1, 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ms = lambda a, w=6, p=3: f"{np.nanmean(a):{w}.{p}f}±{np.nanstd(a):{w - 1}.{p}f}"
    mp = lambda a: ms(a, 5, 1)

    # ---- run everything first (utils_ours / Lightning logs are printed here) ----
    results = []
    for p_c, m in SETTINGS:
        gt = ground_truth(p_c, m)
        runs = [run_setting(p_c, m, s, args.epochs, device) for s in range(args.seeds)]
        results.append((p_c, m, gt, runs))

    # ---- Table A: distribution level, max vs min correction ----
    hA = (f"{'p_c':>5} {'m':>2} | {'%src*':>6} {'%src max':>12} {'%src min':>12} | "
          f"{'Rsrc*':>6} {'Rsrc max':>13} {'Rsrc min':>13} | {'U0*':>6} {'U0 max':>13} {'U0 min':>13} | "
          f"{'U1*':>6} {'U1 max':>13} | {'Rmech*':>6} {'Rmech max':>13} | {'active':>6}")
    linesA, errA = [], {"max": [], "min": []}
    for p_c, m, gt, runs in results:
        dmax = [r[0] for r in runs]; dmin = [r[1] for r in runs]
        fmax = np.array([frac(d["Rsrc"], d["R"]) for d in dmax])
        fmin = np.array([frac(d["Rsrc"], d["R"]) for d in dmin])
        gtf = frac(gt["Rsrc"], gt["R"])
        errA["max"].append(abs(np.nanmean(fmax) - gtf)); errA["min"].append(abs(np.nanmean(fmin) - gtf))
        pc_str = "-" if p_c is None else f"{p_c:.2f}"
        linesA.append(
            f"{pc_str:>5} {m:>2} | {gtf:6.1f} {mp(fmax)} {mp(fmin)} | "
            f"{gt['Rsrc']:6.3f} {ms([d['Rsrc'] for d in dmax])} {ms([d['Rsrc'] for d in dmin])} | "
            f"{gt['U0']:6.3f} {ms([d['U0'] for d in dmax])} {ms([d['U0'] for d in dmin])} | "
            f"{gt['U1']:6.3f} {ms([d['U1'] for d in dmax])} | "
            f"{gt['Rmech']:6.3f} {ms([d['Rmech'] for d in dmax])} | "
            f"{100 * np.mean([d['activated'] for d in dmax]):5.0f}%")

    # ---- Table B: pointwise level ----
    hB = (f"{'p_c':>5} {'m':>2} | {'R*':>6} {'R max':>13} {'R min':>13} | {'Rsrc*':>6} {'Rsrc max':>13} "
          f"{'Rsrc min':>13} {'Rsrc min+sgn':>13} | {'%src*':>6} {'%max':>11} {'%min':>11} {'%min+sgn':>11} | "
          f"{'act.':>5} {'negU max':>8} {'negU min':>8}")
    linesB, errB = [], {"max": [], "min": [], "minsgn": []}
    for p_c, m, gt, runs in results:
        P = [r[2] for r in runs]
        g = lambda k: np.array([p[k] for p in P])
        fmax = 100 * g("Rsrc_max") / g("R_max")
        fmin = 100 * g("Rsrc_min") / g("R_min")
        fsgn = 100 * g("Rsrc_minsgn") / g("R_min")
        gtf = frac(gt["Rsrc"], gt["R"])
        for k, f_ in (("max", fmax), ("min", fmin), ("minsgn", fsgn)):
            errB[k].append(abs(np.nanmean(f_) - gtf))
        pc_str = "-" if p_c is None else f"{p_c:.2f}"
        linesB.append(
            f"{pc_str:>5} {m:>2} | {gt['R']:6.3f} {ms(g('R_max'))} {ms(g('R_min'))} | {gt['Rsrc']:6.3f} "
            f"{ms(g('Rsrc_max'))} {ms(g('Rsrc_min'))} {ms(g('Rsrc_minsgn'))} | {gtf:6.1f} "
            f"{ms(fmax, 5, 1)} {ms(fmin, 5, 1)} {ms(fsgn, 5, 1)} | "
            f"{np.mean(g('activation')):4.0f}% {np.mean(g('negU_max')):7.1f}% {np.mean(g('negU_min')):7.1f}%")

    w = max(len(hA), len(hB))
    report = "\n".join([
        "", "=" * w,
        f"MIXTURE OF SOURCE AND MECHANISTIC REDUNDANCY{' + UNIQUE (' + str(N_UNIQUE) + ' bit)' if N_UNIQUE else ''}"
        f" — lambda = {LAMBDA:g} — max vs min correction ({args.seeds} seeds, {args.epochs} epochs)",
        "* = exact ground truth (nats). Estimates: mean ± std over seeds.",
        "max = current method (R <- max(R, Rsrc));  min = alternative (Rsrc <- min(Rsrc, R), R untouched);",
        "min+sgn = alternative + sign criterion on the two SupGK heads.",
        "=" * w, "",
        "TABLE A — distribution level   (active = % of seeds where Rsrc > R before correction)",
        hA, "-" * len(hA), *linesA, "-" * len(hA),
        f"MAE on %src:  max = {np.nanmean(errA['max']):.1f} pts   min = {np.nanmean(errA['min']):.1f} pts",
        "",
        "TABLE B — pointwise level (means over test samples)",
        "act. = % of samples where r_src > r;  negU = % of samples with a negative unique term",
        hB, "-" * len(hB), *linesB, "-" * len(hB),
        f"MAE on %src:  max = {np.nanmean(errB['max']):.1f} pts   min = {np.nanmean(errB['min']):.1f} pts   "
        f"min+sgn = {np.nanmean(errB['minsgn']):.1f} pts",
        "=" * w, "",
    ])
    print(report)
    tag = (f"_u{N_UNIQUE}" if N_UNIQUE else "") + (f"_lam{LAMBDA:g}" if LAMBDA != 10 else "")
    with open(f"mixture_results{tag}.txt", "w") as fh:
        fh.write(report)
    import pickle
    with open(f"mixture_results{tag}.pkl", "wb") as fh:
        pickle.dump(results, fh)
    print(f"Tables also saved to mixture_results{tag}.txt (raw results in mixture_results{tag}.pkl)")


if __name__ == "__main__":
    main()