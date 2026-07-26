"""
CLIP vs. Supervised Gacs-Korner source redundancy under a shared-but-irrelevant latent.

QUESTION
--------
Source redundancy is the part of redundancy that comes from SHARED INPUTS rather than from
a shared prediction mechanism. A natural way to estimate it is to build a common
representation of the two modalities and read the redundancy off it. CLIP does exactly that,
and (Appendix A) CLIP is a relaxed Gacs-Korner objective. But C_GK maximizes H(Z) among
common factors, IRRESPECTIVE of the target. So if the modalities share something that has
nothing to do with Y, an unsupervised common representation should latch onto it and the
resulting source-redundancy estimate should be wrong -- even though the true PID has not
changed at all. SupGK replaces H(Z) with I(Z;Y) under the same alignment constraint and
should be immune.

This script builds a dataset where that prediction can be tested with the ground truth held
exactly fixed by construction.

CONSTRUCTION -- three disjoint channels per modality
----------------------------------------------------
    base   = tanh(z_c @ Q)              Q orthogonal; shared signal content, D_s dims
    score  = base @ v                   LINEAR readout of what the network observes
    y      = quantile_bins(score)       balanced -> H(Y) = log C

    signal_m   = base @ R_m             R_m orthogonal, different per modality   [D_s]
    nuisance_m = tanh(alpha * zn_m @ Tn_m)                                       [D_n]
    private_m  = tanh(z_p^m @ Tp_m)                                              [D_p]

    x_m = [signal_m | nuisance_m | private_m] + obs_noise

Q, v and R_m come from a generator seeded independently of (d_n, alpha, rho, seed), so the
signal channel is bit-for-bit identical in every condition. Sweeping d_n changes ONLY the
rank of the nuisance common factor; alpha changes only its scale; rho only how much of it
is genuinely shared. None of them touch the label channel.

GROUND TRUTH (constant across the entire sweep)
-----------------------------------------------
y is a deterministic function of `base`, which both modalities carry up to an orthogonal
rotation; z_p lives in its own coordinates. Hence U1 = U2 = S ~ 0 and

    R = I(X1,X2;Y) = H(Y) - H(Y|X1,X2)      and      R_source = R

so the expected source fraction is 100% in every condition. Because y is only two linear
maps from the input (R_m^T then v), I_joint should also approach H(Y) = log C; --sanity
reports the fraction recovered so that Stage-1 underfitting is visible rather than being
mistaken for a property of the data.

We report the source fraction against two denominators:
    src%(imin)  = R_src / R_imin     the paper's own estimator (Eq. 3)
    src%(joint) = R_src / I_joint    ground-truth anchored, immune to a broken i_min

ENTROPY COMPETITION -- the mechanism under test
-----------------------------------------------
Common factors available: `base` (rank D_s = 100) and z_n (rank min(d_n, D_n)). A
max-entropy common representation should prefer whichever has more entropy, so CLIP is
expected to hold onto the signal while d_n < 100 and to lose it as d_n grows past that.
SupGK, maximizing I(Z;Y), should stay flat. The crossover near d_n ~ d_c is the predicted
signature. The disjoint channels make the signal TRIVIALLY accessible, which strengthens
the result: CLIP cannot be said to have missed z_c because finding it was hard.

WHY WE DO NOT PRETRAIN THE ENCODERS WITH A CLASSIFICATION LOSS
---------------------------------------------------------------
The paper's Stage 2 runs on frozen CE-pretrained features. Those features have already
discarded the nuisance, which would erase the effect under test and make the comparison
unfair to CLIP. Here each branch trains trunk + projector end-to-end under its own
objective, with identical architecture, optimizer, batch size and epoch budget. Only the
loss differs.

WHY L_SupGK COLLAPSES FROM SCRATCH, AND THE FIX
------------------------------------------------
With tau = 0.1 and l2-normalized embeddings, random init gives ||z1 - z2||^2 ~ 2, hence
K = exp(-10) ~ 0. Since dK/d(d^2) = -K/(2 tau), BOTH label-aware terms have vanishing
gradient at init, while the alignment penalty contributes lambda * 2 = 20 of a total loss
of ~21. The only descent direction is "align everything", whose cheapest solution is a
constant embedding: perfectly aligned, perfectly uninformative. The paper never hits this
because frozen CE features start with an unsaturated kernel and y-information already
present -- the stop-gradient head is safe there and fatal here. Fixes, both reported so
they can be stated honestly in the paper:
    (i)  tau by the median heuristic, tau = median(d^2)/2 per batch. This is Parzen /
         Rosenblatt bandwidth selection, not a hack: a KDE bandwidth must be scaled to the
         data it is applied to.
    (ii) lambda warm-up over the first `lam_warmup` fraction of training.
Collapse is monitored directly via the effective rank of the embedding spectrum.

USAGE
-----
    python clip_vs_supgk.py --sanity --seeds 2            # RUN THIS FIRST
    python clip_vs_supgk.py --sweep capacity --seeds 3    # answers "just widen the embedding"
    python clip_vs_supgk.py --sweep nuisance --seeds 3    # main result
    python clip_vs_supgk.py --sweep all --seeds 3
    python clip_vs_supgk.py --sweep nuisance --quick --seeds 1   # smoke test only
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = pick_device()
SIGNAL_SEED = 20260724      # signal channel identical in every condition


# ======================================================================================
# CONFIG  (aligned with the paper's synthetic benchmark, App. C.1-C.2 / E)
# ======================================================================================
def default_cfg():
    return dict(
        # ---- data ----
        n=20000, d_c=100, d_p=100, feature_sep=0.5,
        D_s=100, D_n=200, D_p=100,          # disjoint channels -> input dim 400
        num_classes=4, obs_noise=0.02, split=(0.70, 0.15),

        # ---- Stage 1: raw predictors, give I_1, I_2, I_joint, R_imin ----
        s1_hidden=1024, s1_head_hidden=1024,
        s1_lr=1e-4, s1_wd=1e-2, s1_bs=32, s1_epochs=30, s1_clip=8.0,

        # ---- Stage 2: identical across both branches ----
        enc_hidden=1024, proj_hidden=1024, d_emb=512, head_hidden=1024,
        s2_lr=1e-4, s2_bs=512, s2_epochs=100,
        clip_tau=0.1,        # InfoNCE temperature (softmax over cosine; does not saturate)
        gk_tau=None,         # None -> median heuristic per batch; float -> fixed
        lam=10.0, lam_warmup=0.3,

        # ---- probe on frozen embeddings ----
        probe_epochs=60, probe_lr=1e-3, probe_bs=1024,
    )


# ======================================================================================
# 1. DATA
# ======================================================================================
def _orth(rng, n, m):
    """Matrix with orthonormal columns, shape (n, m)."""
    k = max(n, m)
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    return Q[:n, :m]


def make_dataset(cfg, d_n, alpha, rho=1.0, seed=0):
    rng = np.random.default_rng(1000 + seed)
    rng_s = np.random.default_rng(SIGNAL_SEED)      # never depends on d_n/alpha/rho/seed
    n, d_c, d_p = cfg["n"], cfg["d_c"], cfg["d_p"]
    sd = np.sqrt(cfg["feature_sep"])

    z_c = sd * rng.standard_normal((n, d_c))
    z_n = sd * rng.standard_normal((n, d_n)) if d_n > 0 else np.zeros((n, 0))
    if d_n > 0 and rho < 1.0:
        z_n2 = rho * z_n + np.sqrt(1.0 - rho ** 2) * sd * rng.standard_normal((n, d_n))
    else:
        z_n2 = z_n
    z_p = [sd * rng.standard_normal((n, d_p)) for _ in range(2)]

    # ---- shared signal content: well conditioned, fixed across all conditions ----
    Q = _orth(rng_s, d_c, cfg["D_s"])
    base = np.tanh(z_c @ Q)                          # (n, D_s)

    # ---- label: LINEAR readout of the observed signal content ----
    v = rng_s.standard_normal((cfg["D_s"],)) / np.sqrt(cfg["D_s"])
    score = base @ v
    qs = np.quantile(score, np.linspace(0, 1, cfg["num_classes"] + 1)[1:-1])
    y = np.digitize(score, qs).astype(np.int64)

    # ---- each modality sees an orthogonal rotation of the SAME signal vector ----
    R = [_orth(rng_s, cfg["D_s"], cfg["D_s"]) for _ in range(2)]
    sig = [base @ R[m] for m in range(2)]

    # ---- nuisance channel: only rank (d_n) and scale (alpha) vary ----
    if d_n > 0:
        Tn = [rng.standard_normal((d_n, cfg["D_n"])) / np.sqrt(d_n) for _ in range(2)]
        nui = [np.tanh(alpha * (zz @ Tn[m])) for m, zz in enumerate([z_n, z_n2])]
    else:
        nui = [np.zeros((n, cfg["D_n"])) for _ in range(2)]

    # ---- private channel ----
    Tp = [rng.standard_normal((d_p, cfg["D_p"])) / np.sqrt(d_p) for _ in range(2)]
    pri = [np.tanh(z_p[m] @ Tp[m]) for m in range(2)]

    X = []
    for m in range(2):
        x = np.concatenate([sig[m], nui[m], pri[m]], axis=1)
        x = x + cfg["obs_noise"] * rng.standard_normal(x.shape)
        X.append(x.astype(np.float32))

    idx = rng.permutation(n)
    n_tr, n_va = int(cfg["split"][0] * n), int(cfg["split"][1] * n)
    sl = {"train": idx[:n_tr], "valid": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    out = {}
    for split, ii in sl.items():
        out[split] = {
            "0": torch.from_numpy(X[0][ii]), "1": torch.from_numpy(X[1][ii]),
            "label": torch.from_numpy(y[ii]),
            "z_c": torch.from_numpy(z_c[ii].astype(np.float32)),
            "z_n": torch.from_numpy(z_n[ii].astype(np.float32)) if d_n > 0 else None,
        }
    return out


# ======================================================================================
# 2. ARCHITECTURES
# ======================================================================================
class MLP(nn.Module):
    """Two-layer perceptron, matching the paper's MLP shape."""

    def __init__(self, indim, hiddim, outdim):
        super().__init__()
        self.fc, self.fc2 = nn.Linear(indim, hiddim), nn.Linear(hiddim, outdim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc(x)))


class Stage1Predictor(nn.Module):
    """Linear(d_in -> H) + MLP(H -> H -> C). Also used for the joint model."""

    def __init__(self, d_in, hidden, head_hidden, n_classes):
        super().__init__()
        self.enc = nn.Linear(d_in, hidden)
        self.head = MLP(hidden, head_hidden, n_classes)

    def forward(self, x):
        return self.head(self.enc(x))


class Branch(nn.Module):
    """Trunk + two-layer MLP projector -> l2-normalized embedding. Identical per branch."""

    def __init__(self, d_in, enc_hidden, proj_hidden, d_emb):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, enc_hidden), nn.ReLU())
        self.proj = MLP(enc_hidden, proj_hidden, d_emb)

    def forward(self, x):
        return F.normalize(self.proj(self.enc(x)), dim=-1)


# ======================================================================================
# 3. OBJECTIVES
# ======================================================================================
def clip_loss(z1, z2, tau):
    """Symmetric InfoNCE (paper Eq. 21)."""
    logits = z1 @ z2.t() / tau
    tgt = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def supgk_loss(z1, z2, y, tau, lam):
    """L_SupGK (paper Eq. 8). tau=None -> median heuristic (keeps the kernel off its floor)."""
    d2 = torch.cdist(z1, z2, p=2).pow(2)
    if tau is None:
        tau_t = torch.clamp(d2.detach().median() / 2.0, min=1e-3)
        tau_val = float(tau_t.item())
    else:
        tau_t, tau_val = tau, float(tau)
    K = torch.exp(-d2 / (2 * tau_t))
    same = (y[:, None] == y[None, :])
    diff = ~same
    l_pos = ((K[same] - 1.0) ** 2).mean() if same.any() else z1.new_zeros(())
    l_neg = (K[diff] ** 2).mean() if diff.any() else z1.new_zeros(())
    l_align = (z1 - z2).pow(2).sum(-1).mean()
    return l_pos + l_neg + lam * l_align, tau_val


@torch.no_grad()
def embedding_health(Z):
    """Collapse detector: effective rank (entropy of the singular spectrum) and spread."""
    Zc = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Zc.double())
    p = torch.clamp(s / torch.clamp(s.sum(), min=1e-12), min=1e-12)
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    sub = Z[torch.randperm(len(Z))[:2000]]
    return eff_rank, float(torch.cdist(sub, sub).pow(2).mean())


def train_branch(mode, data, cfg):
    """mode in {'clip','supgk'}. Trained from raw inputs; no classification pretraining."""
    d_in = data["train"]["0"].shape[1]
    g1 = Branch(d_in, cfg["enc_hidden"], cfg["proj_hidden"], cfg["d_emb"]).to(DEVICE)
    g2 = Branch(d_in, cfg["enc_hidden"], cfg["proj_hidden"], cfg["d_emb"]).to(DEVICE)
    params = list(g1.parameters()) + list(g2.parameters())

    h_r = None
    if mode == "supgk":
        h_r = MLP(cfg["d_emb"], cfg["head_hidden"], cfg["num_classes"]).to(DEVICE)
        params += list(h_r.parameters())

    opt = torch.optim.Adam(params, lr=cfg["s2_lr"])
    tr = data["train"]
    dl = DataLoader(TensorDataset(tr["0"], tr["1"], tr["label"]),
                    batch_size=cfg["s2_bs"], shuffle=True, drop_last=True)
    n_steps = max(1, cfg["s2_epochs"] * len(dl))
    step, tau_seen = 0, []

    for _ in range(cfg["s2_epochs"]):
        g1.train(); g2.train()
        for x1, x2, y in dl:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            z1, z2 = g1(x1), g2(x2)

            if mode == "clip":
                loss = clip_loss(z1, z2, cfg["clip_tau"])
            else:
                # warm up lambda so the label-aware kernel terms can shape the geometry
                # before the alignment penalty can drive everything to a constant
                w = min(1.0, step / max(1.0, cfg["lam_warmup"] * n_steps))
                loss, tau_used = supgk_loss(z1, z2, y, cfg["gk_tau"], cfg["lam"] * w)
                tau_seen.append(tau_used)
                zm = ((z1 + z2) / 2).detach()          # stop-gradient head (Alg. 1)
                loss = loss + F.cross_entropy(h_r(z1.detach()), y) \
                            + F.cross_entropy(h_r(z2.detach()), y) \
                            + F.cross_entropy(h_r(zm), y)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["s1_clip"])
            opt.step()
            step += 1

    return g1.eval(), g2.eval(), (float(np.mean(tau_seen)) if tau_seen else cfg["clip_tau"])


# ======================================================================================
# 4. MI READOUT
# ======================================================================================
def empirical_log_py(y, num_classes):
    cnt = torch.bincount(y, minlength=num_classes).float()
    return torch.log(torch.clamp(cnt / cnt.sum(), 1e-12, 1.0))


@torch.no_grad()
def embed(g1, g2, split, bs=4096):
    Z1, Z2 = [], []
    for i in range(0, len(split["label"]), bs):
        Z1.append(g1(split["0"][i:i + bs].to(DEVICE)).cpu())
        Z2.append(g2(split["1"][i:i + bs].to(DEVICE)).cpu())
    return torch.cat(Z1), torch.cat(Z2)


@torch.no_grad()
def pointwise_mi_from_head(head, X, y, log_py, bs=4096):
    """i(x;y) = log p(y|x) - log p(y)."""
    outs = []
    for i in range(0, len(y), bs):
        outs.append(F.log_softmax(head(X[i:i + bs].to(DEVICE)), dim=-1).cpu())
    logp = torch.cat(outs)
    return logp[torch.arange(len(y)), y] - log_py[y]


def r_imin(i1, i2):
    """r = max(0, min(i1, i2))  -- paper Eq. 3 / Eq. 9."""
    return torch.clamp(torch.minimum(i1, i2), min=0.0)


def fit_probe(Z_tr, y_tr, cfg):
    """Shared head on FROZEN embeddings, pooled over modalities. Same protocol both branches."""
    head = MLP(cfg["d_emb"], cfg["head_hidden"], cfg["num_classes"]).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=cfg["probe_lr"])
    dl = DataLoader(TensorDataset(Z_tr, y_tr), batch_size=cfg["probe_bs"], shuffle=True)
    loss_v, acc_v = float("nan"), float("nan")
    for _ in range(cfg["probe_epochs"]):
        tot, corr, cnt = 0.0, 0, 0
        for z, y in dl:
            z, y = z.to(DEVICE), y.to(DEVICE)
            out = head(z)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); corr += (out.argmax(1) == y).sum().item(); cnt += len(y)
        loss_v, acc_v = tot / cnt, corr / cnt
    return head.eval(), loss_v, acc_v


def train_stage1(x_tr, y_tr, cfg):
    m = Stage1Predictor(x_tr.shape[1], cfg["s1_hidden"], cfg["s1_head_hidden"],
                        cfg["num_classes"]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["s1_lr"], weight_decay=cfg["s1_wd"])
    dl = DataLoader(TensorDataset(x_tr, y_tr), batch_size=cfg["s1_bs"], shuffle=True)
    for _ in range(cfg["s1_epochs"]):
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["s1_clip"])
            opt.step()
    return m.eval()


@torch.no_grad()
def _accuracy(model, X, y, bs=4096):
    preds = []
    for i in range(0, len(X), bs):
        preds.append(model(X[i:i + bs].to(DEVICE)).argmax(1).cpu())
    return float((torch.cat(preds) == y).float().mean())


def stage1_quantities(data, cfg, log_py):
    """I_1, I_2, I_joint, R_imin (nats) plus joint accuracy and the synergy leak."""
    tr, te = data["train"], data["test"]
    h1 = train_stage1(tr["0"], tr["label"], cfg)
    h2 = train_stage1(tr["1"], tr["label"], cfg)
    Xtr_j, Xte_j = torch.cat([tr["0"], tr["1"]], 1), torch.cat([te["0"], te["1"]], 1)
    hj = train_stage1(Xtr_j, tr["label"], cfg)

    i1 = pointwise_mi_from_head(h1, te["0"], te["label"], log_py)
    i2 = pointwise_mi_from_head(h2, te["1"], te["label"], log_py)
    ij = pointwise_mi_from_head(hj, Xte_j, te["label"], log_py)
    r = r_imin(i1, i2)
    R = r.mean().item()

    q = dict(I_1=i1.mean().item(), I_2=i2.mean().item(),
             I_joint=ij.mean().item(), R_imin=R,
             acc_1=_accuracy(h1, te["0"], te["label"]),
             acc_j=_accuracy(hj, Xte_j, te["label"]))
    # S = I_joint - U1 - U2 - R with Um = Im - R
    q["S_leak"] = q["I_joint"] - q["I_1"] - q["I_2"] + R
    return q, r


# ======================================================================================
# 5. MECHANISM DIAGNOSTIC -- what does the shared space actually encode?
# ======================================================================================
def ridge_r2(Z_tr, Z_te, Y_tr, Y_te, ridge=1e-3):
    Xtr = torch.cat([Z_tr, torch.ones(len(Z_tr), 1)], 1).numpy().astype(np.float64)
    Xte = torch.cat([Z_te, torch.ones(len(Z_te), 1)], 1).numpy().astype(np.float64)
    Ytr, Yte = Y_tr.numpy().astype(np.float64), Y_te.numpy().astype(np.float64)
    A = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    res = ((Yte - Xte @ W) ** 2).sum()
    tot = ((Yte - Yte.mean(0)) ** 2).sum()
    return float(1 - res / tot)


# ======================================================================================
# 6. ONE CONDITION
# ======================================================================================
def run_condition(cfg, d_n, alpha, rho, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    data = make_dataset(cfg, d_n=d_n, alpha=alpha, rho=rho, seed=seed)
    tr, te = data["train"], data["test"]
    log_py = empirical_log_py(tr["label"], cfg["num_classes"])
    H_Y = float(-(torch.exp(log_py) * log_py).sum())

    s1, r_vec = stage1_quantities(data, cfg, log_py)
    row = dict(d_n=d_n, alpha=alpha, rho=rho, d_emb=cfg["d_emb"], seed=seed,
               H_Y=H_Y, frac_HY=100.0 * s1["I_joint"] / max(H_Y, 1e-8), **s1)

    for mode in ("clip", "supgk"):
        g1, g2, tau_used = train_branch(mode, data, cfg)
        Z1_tr, Z2_tr = embed(g1, g2, tr)
        Z1_te, Z2_te = embed(g1, g2, te)

        head, ploss, pacc = fit_probe(torch.cat([Z1_tr, Z2_tr]),
                                      torch.cat([tr["label"], tr["label"]]), cfg)
        i1 = pointwise_mi_from_head(head, Z1_te, te["label"], log_py)
        i2 = pointwise_mi_from_head(head, Z2_te, te["label"], log_py)
        r_gk = r_imin(i1, i2)
        R_src = r_gk.mean().item()
        eff_rank, mean_d2 = embedding_health(Z1_te)

        row.update({
            f"R_src_{mode}": R_src,
            f"src_imin_{mode}": 100.0 * R_src / max(s1["R_imin"], 1e-8),
            f"src_joint_{mode}": 100.0 * R_src / max(s1["I_joint"], 1e-8),
            f"probe_loss_{mode}": ploss, f"probe_acc_{mode}": pacc,
            f"align_{mode}": (Z1_te - Z2_te).pow(2).sum(-1).mean().item(),
            f"effrank_{mode}": eff_rank, f"meand2_{mode}": mean_d2, f"tau_{mode}": tau_used,
            f"gk_exceeds_{mode}": float((r_gk > r_vec).float().mean()),
            f"r2_zc_{mode}": ridge_r2(Z1_tr, Z1_te, tr["z_c"], te["z_c"]),
            f"r2_zn_{mode}": (ridge_r2(Z1_tr, Z1_te, tr["z_n"], te["z_n"])
                              if d_n > 0 else float("nan")),
        })
    return row


# ======================================================================================
# 7. SWEEPS
# ======================================================================================
def build_grid(name, cfg):
    """Each entry: (d_n, alpha, rho, d_emb)."""
    de = cfg["d_emb"]
    if name == "nuisance":
        return [(0, 0.0, 1.0, de)] + [(d, 1.0, 1.0, de) for d in (25, 50, 100, 200, 400)]
    if name == "capacity":
        return [(200, 1.0, 1.0, d) for d in (32, 64, 128, 256, 512)]
    if name == "alpha":
        return [(200, a, 1.0, de) for a in (0.0, 0.25, 0.5, 1.0, 2.0)]
    if name == "rho":
        return [(200, 1.0, r, de) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    raise ValueError(name)


def sanity(cfg, seeds):
    print("\nSANITY: ground truth must be FLAT in d_n, and I_joint should approach H(Y)")
    print(f"{'d_n':>6}{'H(Y)':>8}{'I_1':>8}{'I_2':>8}{'I_joint':>10}"
          f"{'%H(Y)':>8}{'acc_1':>8}{'acc_j':>8}{'R_imin':>9}{'S_leak':>9}")
    print("-" * 82)
    for d_n in (0, 50, 100, 200, 400):
        acc = []
        for s in range(seeds):
            torch.manual_seed(s); np.random.seed(s)
            data = make_dataset(cfg, d_n=d_n, alpha=1.0, rho=1.0, seed=s)
            lp = empirical_log_py(data["train"]["label"], cfg["num_classes"])
            q, _ = stage1_quantities(data, cfg, lp)
            q["H_Y"] = float(-(torch.exp(lp) * lp).sum())
            q["frac"] = 100.0 * q["I_joint"] / max(q["H_Y"], 1e-8)
            acc.append(q)
        f = lambda k: np.mean([a[k] for a in acc])
        print(f"{d_n:6d}{f('H_Y'):8.3f}{f('I_1'):8.3f}{f('I_2'):8.3f}{f('I_joint'):10.3f}"
              f"{f('frac'):8.1f}{f('acc_1'):8.2f}{f('acc_j'):8.2f}"
              f"{f('R_imin'):9.3f}{f('S_leak'):9.3f}")
    print("-" * 82)
    print("PASS if: every column flat across rows (within seed noise);")
    print("         I_joint ~ I_1 ~ I_2 ~ R_imin;  %H(Y) > 80 and acc_j > 0.90;")
    print("         |S_leak| small relative to R_imin.")
    print("Flat but low %H(Y)  -> Stage 1 underfits: raise s1_epochs, drop --quick.")
    print("Drifting columns    -> the signal channel is still affected by d_n: STOP.")
    print("Large S_leak        -> lower --obs-noise.\n")


# ======================================================================================
# 8. MAIN
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="nuisance",
                    choices=["nuisance", "capacity", "alpha", "rho", "all"])
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gk-tau", type=float, default=None,
                    help="fixed SupGK bandwidth; omit for the median heuristic")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--obs-noise", type=float, default=None)
    ap.add_argument("--s1-epochs", type=int, default=None)
    ap.add_argument("--out", default="clip_vs_supgk_results.json")
    args = ap.parse_args()

    cfg = default_cfg()
    for key, val in [("gk_tau", args.gk_tau), ("lam", args.lam),
                     ("obs_noise", args.obs_noise), ("s1_epochs", args.s1_epochs)]:
        if val is not None:
            cfg[key] = val
    if args.quick:
        # scale BATCH SIZE with n so the number of gradient steps stays sane
        cfg.update(n=6000, s1_epochs=10, s1_bs=64, s2_bs=128, s2_epochs=30, probe_epochs=25)

    print(f"device: {DEVICE}")
    if args.sanity:
        sanity(cfg, args.seeds)
        return

    sweeps = ["nuisance", "capacity", "alpha", "rho"] if args.sweep == "all" else [args.sweep]
    rows = []

    for sw in sweeps:
        print(f"\n{'#' * 100}\n# sweep: {sw}\n{'#' * 100}")
        for (d_n, alpha, rho, d_emb) in build_grid(sw, cfg):
            c = dict(cfg); c["d_emb"] = d_emb
            for seed in range(args.seeds):
                r = run_condition(c, d_n, alpha, rho, seed)
                r["sweep"] = sw
                rows.append(r)
                zn = ("n/a " if np.isnan(r["r2_zn_clip"])
                      else f"{r['r2_zn_clip']:.2f}/{r['r2_zn_supgk']:.2f}")
                print(f"[{sw}] d_n={d_n:4d} a={alpha:.2f} rho={rho:.2f} emb={d_emb:4d} s={seed}\n"
                      f"    H(Y)={r['H_Y']:.3f} I_1={r['I_1']:.3f} I_2={r['I_2']:.3f} "
                      f"I_joint={r['I_joint']:.3f} ({r['frac_HY']:.0f}% of H(Y)) "
                      f"R_imin={r['R_imin']:.3f} S_leak={r['S_leak']:+.3f}\n"
                      f"    R_src CLIP={r['R_src_clip']:.3f} SupGK={r['R_src_supgk']:.3f}   "
                      f"src%(joint) CLIP={r['src_joint_clip']:6.1f} "
                      f"SupGK={r['src_joint_supgk']:6.1f}\n"
                      f"    probe_acc {r['probe_acc_clip']:.2f}/{r['probe_acc_supgk']:.2f}  "
                      f"effrank {r['effrank_clip']:6.1f}/{r['effrank_supgk']:6.1f}  "
                      f"align {r['align_clip']:.3f}/{r['align_supgk']:.3f}  "
                      f"tau_gk={r['tau_supgk']:.3f}\n"
                      f"    R2(z_c) {r['r2_zc_clip']:.2f}/{r['r2_zc_supgk']:.2f}  "
                      f"R2(z_n) {zn}")

        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)

        print(f"\n{'-' * 112}")
        print(f"{'d_n':>5}{'alpha':>7}{'rho':>6}{'emb':>6}{'I_joint':>9}{'R_imin':>8}"
              f"{'src%J CLIP':>16}{'src%J SupGK':>16}{'effrk C/G':>17}")
        print("-" * 112)
        for (d_n, alpha, rho, d_emb) in build_grid(sw, cfg):
            sel = [r for r in rows if r["sweep"] == sw and r["d_n"] == d_n and
                   r["alpha"] == alpha and r["rho"] == rho and r["d_emb"] == d_emb]
            if not sel:
                continue
            m = lambda k: np.mean([s[k] for s in sel])
            sd = lambda k: np.std([s[k] for s in sel])
            print(f"{d_n:5d}{alpha:7.2f}{rho:6.2f}{d_emb:6d}"
                  f"{m('I_joint'):9.3f}{m('R_imin'):8.3f}"
                  f"{m('src_joint_clip'):10.1f} ± {sd('src_joint_clip'):3.1f}"
                  f"{m('src_joint_supgk'):10.1f} ± {sd('src_joint_supgk'):3.1f}"
                  f"{m('effrank_clip'):9.0f}/{m('effrank_supgk'):<7.0f}")
        print("-" * 112)
        print("Ground truth every row: R = I_joint, 100% source.")
        print("Expect SupGK flat near 100; CLIP falling as d_n grows past d_c = 100.")
        print("effrank near 1 = collapsed embedding, so the number beside it is meaningless.")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()