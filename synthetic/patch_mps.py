"""
CLIP vs. Supervised Gacs-Korner for estimating SOURCE redundancy.

CLAIM UNDER TEST
    C_GK maximizes H(Z) among common factors, irrespective of Y. If two modalities share
    something unrelated to the target, an unsupervised common representation (CLIP is a
    relaxed GK objective, paper App. A) should latch onto it and misestimate source
    redundancy -- even though the true PID has not moved. SupGK maximizes I(Z;Y) under the
    same alignment constraint and should be immune.

CONSTRUCTION -- three disjoint channels per modality (input dim 400)
    base  = tanh(z_c @ Q)                 Q orthogonal, shared signal content   [D_s=100]
    y     = quantile_bins(base @ v)       LINEAR readout -> learnable ceiling
    sig_m = base @ R_m                    R_m orthogonal, per modality          [D_s]
    nui_m = tanh(alpha * zn_m @ Tn_m)     shared with correlation rho           [D_n=200]
    pri_m = tanh(z_p^m @ Tp_m)            private                               [D_p=100]
    x_m   = [sig_m | nui_m | pri_m] + obs_noise
    Q, v, R_m come from a FIXED generator: the signal channel is bit-for-bit identical in
    every condition. d_n sets the nuisance RANK, alpha its SCALE, rho how much is SHARED.

GROUND TRUTH, constant across every condition
    U1 = U2 = S ~ 0, R = I(X1,X2;Y) = H(Y) - H(Y|X1,X2), and R_source = R.
    -> expected source fraction = 100% everywhere.

READING THE RESULT
    The comparison is CLIP vs SupGK at matched (d_n, alpha, rho, d_emb, seed): identical
    data, architecture, optimizer, budget and probe protocol; only the Stage-2 loss differs.
    Every grid includes an ANCHOR (d_n=0 or rho=0) where no shared-but-irrelevant factor
    exists. Both branches must sit near 100% there. Without that anchor a low CLIP number
    only shows "unsupervised < supervised", which is trivially true and proves nothing.

COMPUTE SAVING
    Datasets are built once per (d_n, alpha, rho, seed) and kept on device (single-entry
    cache). Stage 1 does not depend on d_emb, so its results are cached and reused across
    the whole capacity sweep. Jobs are ordered so d_emb varies fastest and the caches hit.

USAGE
    python clip_vs_supgk.py --sanity --seeds 3                  # run first
    python clip_vs_supgk.py --sweep capacity --seeds 3 --out capacity.json
    python clip_vs_supgk.py --sweep rho      --seeds 3 --out rho.json
    python clip_vs_supgk.py --sweep nuisance --seeds 3 --out nuisance.json
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNAL_SEED = 20260724
PROBE_DIM = 10          # latents probed through a fixed 10-d subspace -> equal R2 ceilings


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def default_cfg():
    return dict(
        n=20000, d_c=100, d_p=100, feature_sep=0.5,
        D_s=100, D_n=200, D_p=100, num_classes=4, obs_noise=0.02, split=(0.70, 0.15),
        # Stage 1 (cached, so its cost amortises: train it properly)
        s1_hidden=1024, s1_head_hidden=1024,
        s1_lr=1e-4, s1_wd=1e-2, s1_bs=32, s1_epochs=50, s1_clip=8.0,
        # Stage 2, identical for both branches
        enc_hidden=1024, proj_hidden=1024, d_emb=512, head_hidden=1024,
        s2_lr=1e-4, s2_bs=512, s2_epochs=100,
        clip_tau=0.1, gk_tau=None, lam=10.0, lam_warmup=0.3,
        # probe
        probe_epochs=60, probe_lr=1e-3, probe_bs=1024,
    )


# ======================================================================================
# DATA
# ======================================================================================
def _orth(rng, n, m):
    k = max(n, m)
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    return Q[:n, :m]


def _build_dataset(cfg, d_n, alpha, rho, seed):
    rng = np.random.default_rng(1000 + seed)
    rng_s = np.random.default_rng(SIGNAL_SEED)
    n, d_c, d_p = cfg["n"], cfg["d_c"], cfg["d_p"]
    sd = np.sqrt(cfg["feature_sep"])

    z_c = sd * rng.standard_normal((n, d_c))
    z_n = sd * rng.standard_normal((n, d_n)) if d_n > 0 else np.zeros((n, 0))
    z_n2 = (rho * z_n + np.sqrt(1 - rho ** 2) * sd * rng.standard_normal((n, d_n))
            if d_n > 0 and rho < 1.0 else z_n)
    z_p = [sd * rng.standard_normal((n, d_p)) for _ in range(2)]

    Q = _orth(rng_s, d_c, cfg["D_s"])
    base = np.tanh(z_c @ Q)
    v = rng_s.standard_normal((cfg["D_s"],)) / np.sqrt(cfg["D_s"])
    score = base @ v
    qs = np.quantile(score, np.linspace(0, 1, cfg["num_classes"] + 1)[1:-1])
    y = np.digitize(score, qs).astype(np.int64)

    R = [_orth(rng_s, cfg["D_s"], cfg["D_s"]) for _ in range(2)]
    sig = [base @ R[m] for m in range(2)]
    if d_n > 0:
        Tn = [rng.standard_normal((d_n, cfg["D_n"])) / np.sqrt(d_n) for _ in range(2)]
        nui = [np.tanh(alpha * (zz @ Tn[m])) for m, zz in enumerate([z_n, z_n2])]
    else:
        nui = [np.zeros((n, cfg["D_n"])) for _ in range(2)]
    Tp = [rng.standard_normal((d_p, cfg["D_p"])) / np.sqrt(d_p) for _ in range(2)]
    pri = [np.tanh(z_p[m] @ Tp[m]) for m in range(2)]

    X = [np.concatenate([sig[m], nui[m], pri[m]], 1).astype(np.float32) for m in range(2)]
    X = [x + cfg["obs_noise"] * rng.standard_normal(x.shape).astype(np.float32) for x in X]

    # fixed low-dim probe targets so R2(z_c) and R2(z_n) share the same ceiling
    Pc = rng_s.standard_normal((d_c, PROBE_DIM)) / np.sqrt(d_c)
    zc_p = (z_c @ Pc).astype(np.float32)
    if d_n > 0:
        Pn = np.random.default_rng(SIGNAL_SEED + 1).standard_normal((d_n, PROBE_DIM))
        zn_p = (z_n @ (Pn / np.sqrt(d_n))).astype(np.float32)
    else:
        zn_p = None

    idx = rng.permutation(n)
    n_tr, n_va = int(cfg["split"][0] * n), int(cfg["split"][1] * n)
    cuts = {"train": idx[:n_tr], "valid": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    out = {}
    for sp, ii in cuts.items():
        d = {"0": torch.from_numpy(X[0][ii]).to(DEVICE),
             "1": torch.from_numpy(X[1][ii]).to(DEVICE),
             "label": torch.from_numpy(y[ii]).to(DEVICE),
             "zc_p": torch.from_numpy(zc_p[ii]),
             "zn_p": torch.from_numpy(zn_p[ii]) if zn_p is not None else None}
        d["joint"] = torch.cat([d["0"], d["1"]], 1)
        out[sp] = d
    return out


_DATA_CACHE = {}      # single entry: a dataset is ~200 MB on device


def get_dataset(cfg, d_n, alpha, rho, seed):
    key = (d_n, alpha, rho, seed, cfg["n"], cfg["obs_noise"])
    if key not in _DATA_CACHE:
        _DATA_CACHE.clear()
        seed_all(seed)
        _DATA_CACHE[key] = _build_dataset(cfg, d_n, alpha, rho, seed)
    return _DATA_CACHE[key]


def batches(n, bs, shuffle=True, drop_last=False):
    idx = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
    end = n - (n % bs) if drop_last else n
    for i in range(0, end, bs):
        yield idx[i:i + bs]


# ======================================================================================
# MODELS / LOSSES
# ======================================================================================
class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.fc, self.fc2 = nn.Linear(i, h), nn.Linear(h, o)

    def forward(self, x):
        return self.fc2(F.relu(self.fc(x)))


class Stage1(nn.Module):
    def __init__(self, d, h, hh, c):
        super().__init__(); self.enc, self.head = nn.Linear(d, h), MLP(h, hh, c)

    def forward(self, x):
        return self.head(self.enc(x))


class Branch(nn.Module):
    def __init__(self, d, eh, ph, de):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, eh), nn.ReLU())
        self.proj = MLP(eh, ph, de)

    def forward(self, x):
        return F.normalize(self.proj(self.enc(x)), dim=-1)


def safe_cdist2(a, b):
    """||a_i-b_j||^2 via matmul only: MPS-safe, avoids any cdist backward fallback."""
    a2 = a.pow(2).sum(-1, keepdim=True)
    b2 = b.pow(2).sum(-1, keepdim=True).t()
    return torch.clamp(a2 + b2 - 2.0 * (a @ b.t()), min=0.0)


def clip_loss(z1, z2, tau):
    lg = z1 @ z2.t() / tau
    t = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(lg, t) + F.cross_entropy(lg.t(), t))


def supgk_loss(z1, z2, y, tau, lam):
    """Paper Eq. 8. tau=None -> median heuristic (keeps the kernel off its floor)."""
    d2 = safe_cdist2(z1, z2)
    if tau is None:
        tt = torch.clamp(d2.detach().median() / 2.0, min=1e-3); tv = float(tt)
    else:
        tt, tv = tau, float(tau)
    K = torch.exp(-d2 / (2 * tt))
    same = y[:, None] == y[None, :]
    loss = ((K[same] - 1.) ** 2).mean() + (K[~same] ** 2).mean() \
        + lam * (z1 - z2).pow(2).sum(-1).mean()
    return loss, tv


# ======================================================================================
# STAGE 1  (validation checkpointing + caching)
# ======================================================================================
@torch.no_grad()
def eval_loss_acc(model, X, y, bs=8192):
    model.eval(); tot, corr = 0.0, 0
    for i in range(0, len(y), bs):
        o = model(X[i:i + bs]); yb = y[i:i + bs]
        tot += F.cross_entropy(o, yb, reduction="sum").item()
        corr += (o.argmax(1) == yb).sum().item()
    return tot / len(y), corr / len(y)


def train_stage1(Xtr, ytr, Xva, yva, cfg):
    m = Stage1(Xtr.shape[1], cfg["s1_hidden"], cfg["s1_head_hidden"],
               cfg["num_classes"]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["s1_lr"], weight_decay=cfg["s1_wd"])
    best, state = float("inf"), None
    for _ in range(cfg["s1_epochs"]):
        m.train()
        for b in batches(len(ytr), cfg["s1_bs"]):
            loss = F.cross_entropy(m(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), cfg["s1_clip"]); opt.step()
        vl, _ = eval_loss_acc(m, Xva, yva)
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in m.state_dict().items()}
    if state:
        m.load_state_dict(state)
    return m.eval()


def empirical_log_py(y, c):
    cnt = torch.bincount(y, minlength=c).float()
    return torch.log(torch.clamp(cnt / cnt.sum(), 1e-12, 1.0))


@torch.no_grad()
def pointwise_mi(model, X, y, log_py, bs=8192):
    """i(x;y) = log p(y|x) - log p(y), per sample, on CPU."""
    lp = torch.cat([F.log_softmax(model(X[i:i + bs]), -1).cpu()
                    for i in range(0, len(y), bs)])
    yc = y.cpu()
    return lp[torch.arange(len(yc)), yc] - log_py.cpu()[yc]


def r_imin(i1, i2):
    return torch.clamp(torch.minimum(i1, i2), min=0.0)


_S1_CACHE = {}        # Stage 1 is independent of d_emb -> reused across the capacity sweep


def stage1_quantities(cfg, d_n, alpha, rho, seed):
    key = (d_n, alpha, rho, seed, cfg["s1_epochs"], cfg["s1_bs"], cfg["n"])
    if key in _S1_CACHE:
        return _S1_CACHE[key]

    data = get_dataset(cfg, d_n, alpha, rho, seed)
    tr, va, te = data["train"], data["valid"], data["test"]
    log_py = empirical_log_py(tr["label"], cfg["num_classes"])

    h1 = train_stage1(tr["0"], tr["label"], va["0"], va["label"], cfg)
    h2 = train_stage1(tr["1"], tr["label"], va["1"], va["label"], cfg)
    hj = train_stage1(tr["joint"], tr["label"], va["joint"], va["label"], cfg)

    i1 = pointwise_mi(h1, te["0"], te["label"], log_py)
    i2 = pointwise_mi(h2, te["1"], te["label"], log_py)
    ij = pointwise_mi(hj, te["joint"], te["label"], log_py)
    r = r_imin(i1, i2); R = r.mean().item()

    q = dict(I_1=i1.mean().item(), I_2=i2.mean().item(), I_joint=ij.mean().item(),
             R_imin=R, acc_1=eval_loss_acc(h1, te["0"], te["label"])[1],
             acc_j=eval_loss_acc(hj, te["joint"], te["label"])[1])
    q["S_leak"] = q["I_joint"] - q["I_1"] - q["I_2"] + R
    q["asym"] = abs(q["I_1"] - q["I_2"])       # X1,X2 interchangeable -> should be ~0
    _S1_CACHE[key] = (q, r, log_py)
    return _S1_CACHE[key]


# ======================================================================================
# STAGE 2
# ======================================================================================
def train_branch(mode, data, cfg):
    d_in = data["train"]["0"].shape[1]
    g1 = Branch(d_in, cfg["enc_hidden"], cfg["proj_hidden"], cfg["d_emb"]).to(DEVICE)
    g2 = Branch(d_in, cfg["enc_hidden"], cfg["proj_hidden"], cfg["d_emb"]).to(DEVICE)
    params = list(g1.parameters()) + list(g2.parameters())
    h_r = None
    if mode == "supgk":
        h_r = MLP(cfg["d_emb"], cfg["head_hidden"], cfg["num_classes"]).to(DEVICE)
        params += list(h_r.parameters())

    opt = torch.optim.Adam(params, lr=cfg["s2_lr"])
    tr = data["train"]; X1, X2, Y = tr["0"], tr["1"], tr["label"]; n = len(Y)
    n_steps = max(1, cfg["s2_epochs"] * (n // cfg["s2_bs"]))
    step, taus = 0, []

    for _ in range(cfg["s2_epochs"]):
        g1.train(); g2.train()
        for b in batches(n, cfg["s2_bs"], drop_last=True):
            z1, z2 = g1(X1[b]), g2(X2[b])
            if mode == "clip":
                loss = clip_loss(z1, z2, cfg["clip_tau"])
            else:
                w = min(1.0, step / max(1.0, cfg["lam_warmup"] * n_steps))
                loss, tv = supgk_loss(z1, z2, Y[b], cfg["gk_tau"], cfg["lam"] * w)
                taus.append(tv)
                zm = ((z1 + z2) / 2).detach()          # stop-gradient head (Alg. 1)
                loss = loss + F.cross_entropy(h_r(z1.detach()), Y[b]) \
                            + F.cross_entropy(h_r(z2.detach()), Y[b]) \
                            + F.cross_entropy(h_r(zm), Y[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, cfg["s1_clip"]); opt.step(); step += 1
    return g1.eval(), g2.eval(), (float(np.mean(taus)) if taus else cfg["clip_tau"])


@torch.no_grad()
def embed(g1, g2, sp, bs=8192):
    n = len(sp["label"])
    return (torch.cat([g1(sp["0"][i:i + bs]) for i in range(0, n, bs)]),
            torch.cat([g2(sp["1"][i:i + bs]) for i in range(0, n, bs)]))


def fit_probe(Ztr, ytr, Zva, yva, cfg):
    """Shared head on FROZEN embeddings, pooled over modalities. Same protocol per branch."""
    head = MLP(cfg["d_emb"], cfg["head_hidden"], cfg["num_classes"]).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=cfg["probe_lr"])
    best, state = float("inf"), None
    for _ in range(cfg["probe_epochs"]):
        head.train()
        for b in batches(len(ytr), cfg["probe_bs"]):
            loss = F.cross_entropy(head(Ztr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        vl, _ = eval_loss_acc(head, Zva, yva)
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in head.state_dict().items()}
    if state:
        head.load_state_dict(state)
    return head.eval()


@torch.no_grad()
def effective_rank(Z):
    """~1 = collapsed to a point; ~num_classes = a class code (SupGK's intended optimum)."""
    Zc = (Z - Z.mean(0, keepdim=True)).cpu().double()      # MPS has no float64
    s = torch.linalg.svdvals(Zc)
    p = torch.clamp(s / torch.clamp(s.sum(), min=1e-12), min=1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def ridge_r2(Ztr, Zte, Ytr, Yte, ridge=1e-3):
    A = np.concatenate([Ztr.cpu().numpy(), np.ones((len(Ztr), 1))], 1).astype(np.float64)
    B = np.concatenate([Zte.cpu().numpy(), np.ones((len(Zte), 1))], 1).astype(np.float64)
    Ytr, Yte = Ytr.numpy().astype(np.float64), Yte.numpy().astype(np.float64)
    W = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ Ytr)
    return float(1 - ((Yte - B @ W) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())


# ======================================================================================
# ONE CONDITION
# ======================================================================================
def run_condition(cfg, d_n, alpha, rho, seed):
    s1, r_vec, log_py = stage1_quantities(cfg, d_n, alpha, rho, seed)
    data = get_dataset(cfg, d_n, alpha, rho, seed)
    tr, va, te = data["train"], data["valid"], data["test"]
    H_Y = float(-(torch.exp(log_py) * log_py).sum())

    row = dict(d_n=d_n, alpha=alpha, rho=rho, d_emb=cfg["d_emb"], seed=seed, H_Y=H_Y,
               frac_HY=100 * s1["I_joint"] / max(H_Y, 1e-8), **s1)

    for mode in ("clip", "supgk"):
        seed_all(seed + 7919)                      # identical init for both branches
        g1, g2, tau = train_branch(mode, data, cfg)
        Z1tr, Z2tr = embed(g1, g2, tr)
        Z1va, Z2va = embed(g1, g2, va)
        Z1te, Z2te = embed(g1, g2, te)
        head = fit_probe(torch.cat([Z1tr, Z2tr]), torch.cat([tr["label"]] * 2),
                         torch.cat([Z1va, Z2va]), torch.cat([va["label"]] * 2), cfg)

        i1 = pointwise_mi(head, Z1te, te["label"], log_py)
        i2 = pointwise_mi(head, Z2te, te["label"], log_py)
        r_gk = r_imin(i1, i2); R_src = r_gk.mean().item()

        row.update({
            f"R_src_{mode}": R_src,
            f"src_imin_{mode}": 100 * R_src / max(s1["R_imin"], 1e-8),
            f"src_joint_{mode}": 100 * R_src / max(s1["I_joint"], 1e-8),
            f"probe_acc_{mode}": eval_loss_acc(head, Z1te, te["label"])[1],
            f"align_{mode}": (Z1te - Z2te).pow(2).sum(-1).mean().item(),
            f"effrank_{mode}": effective_rank(Z1te), f"tau_{mode}": tau,
            f"gk_exceeds_{mode}": float((r_gk > r_vec).float().mean()),
            f"r2_zc_{mode}": ridge_r2(Z1tr, Z1te, tr["zc_p"], te["zc_p"]),
            f"r2_zn_{mode}": (ridge_r2(Z1tr, Z1te, tr["zn_p"], te["zn_p"])
                              if tr["zn_p"] is not None else float("nan")),
        })
    return row


# ======================================================================================
# GRIDS -- every grid carries an ANCHOR with no shared nuisance
# ======================================================================================
def build_grid(name, cfg):
    de = cfg["d_emb"]
    return {
        # anchor d_n=0 at EVERY width: separates "nuisance captured Z" from "unsup < sup"
        "capacity": [(dn, 1.0 if dn else 0.0, 1.0, w)
                     for w in (32, 64, 128, 256, 512, 1024, 2048, 4098) for dn in (0, 200)],
        # cleanest sweep: input marginals identical for every rho, only sharing changes
        "rho":      [(200, 1.0, r, de) for r in (0.0, 0.25, 0.5, 0.75, 1.0)],
        "nuisance": [(0, 0.0, 1.0, de)] + [(d, 1.0, 1.0, de)
                                           for d in (25, 50, 100, 200, 400)],
        "alpha":    [(200, a, 1.0, de) for a in (0.0, 0.25, 0.5, 1.0, 2.0)],
    }[name]


def is_anchor(d_n, alpha, rho):
    return d_n == 0 or rho == 0.0 or alpha == 0.0


def sanity(cfg, seeds):
    print("\nSANITY: ground truth must be FLAT in d_n; I_joint should approach H(Y)")
    print(f"{'d_n':>5}{'H(Y)':>7}{'I_1':>13}{'I_2':>13}{'I_joint':>13}"
          f"{'%H(Y)':>7}{'acc_j':>7}{'R_imin':>8}{'S_leak':>8}{'asym':>7}")
    print("-" * 94)
    worst = 0.0
    for d_n in (0, 50, 100, 200, 400):
        acc = []
        for s in range(seeds):
            q, _, lp = stage1_quantities(cfg, d_n, 1.0, 1.0, s)
            q = dict(q); q["H_Y"] = float(-(torch.exp(lp) * lp).sum())
            q["frac"] = 100 * q["I_joint"] / max(q["H_Y"], 1e-8)
            acc.append(q)
        m = lambda k: np.mean([a[k] for a in acc]); sd = lambda k: np.std([a[k] for a in acc])
        worst = max(worst, m("asym"))
        print(f"{d_n:5d}{m('H_Y'):7.3f}{m('I_1'):8.3f}±{sd('I_1'):4.3f}"
              f"{m('I_2'):8.3f}±{sd('I_2'):4.3f}{m('I_joint'):8.3f}±{sd('I_joint'):4.3f}"
              f"{m('frac'):7.1f}{m('acc_j'):7.2f}{m('R_imin'):8.3f}"
              f"{m('S_leak'):8.3f}{m('asym'):7.3f}")
    print("-" * 94)
    print("PASS: columns flat; I_joint ~ I_1 ~ I_2 ~ R_imin; %H(Y) > 90; |S_leak| small;")
    print("      asym < 0.05 everywhere (X1,X2 interchangeable -> asym is training noise).")
    print(f"\n  {'OK' if worst < 0.05 else 'WARNING: Stage 1 unstable'}: "
          f"worst asym = {worst:.3f}\n")


def summarize(rows, sw, cfg):
    print(f"\n{'-' * 112}")
    print(f"{'d_n':>5}{'rho':>6}{'alpha':>6}{'emb':>5}{'I_joint':>9}{'acc_j':>7}"
          f"{'src%J CLIP':>15}{'src%J SupGK':>15}{'R2zc C/G':>13}{'R2zn C/G':>13}")
    print("-" * 112)
    for (d_n, alpha, rho, de) in build_grid(sw, cfg):
        sel = [r for r in rows if r["sweep"] == sw and r["d_n"] == d_n and
               r["alpha"] == alpha and r["rho"] == rho and r["d_emb"] == de]
        if not sel:
            continue
        m = lambda k: np.mean([s[k] for s in sel]); sd = lambda k: np.std([s[k] for s in sel])
        zn = ("n/a" if np.isnan(m("r2_zn_clip"))
              else f"{m('r2_zn_clip'):.2f}/{m('r2_zn_supgk'):.2f}")
        tag = "  <- ANCHOR" if is_anchor(d_n, alpha, rho) else ""
        print(f"{d_n:5d}{rho:6.2f}{alpha:6.2f}{de:5d}{m('I_joint'):9.3f}{m('acc_j'):7.2f}"
              f"{m('src_joint_clip'):9.1f}±{sd('src_joint_clip'):4.1f}"
              f"{m('src_joint_supgk'):9.1f}±{sd('src_joint_supgk'):4.1f}"
              f"{m('r2_zc_clip'):8.2f}/{m('r2_zc_supgk'):<4.2f}{zn:>13}{tag}")
    print("-" * 112)
    print("Ground truth: R = I_joint, 100% source, in EVERY row.")
    print("Result holds if BOTH branches are ~100 on the ANCHOR rows and CLIP alone falls")
    print("off them. If CLIP is low on the anchor too, the experiment shows nothing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="rho",
                    choices=["capacity", "rho", "nuisance", "alpha", "all"])
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gk-tau", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--s1-epochs", type=int, default=None)
    ap.add_argument("--s2-epochs", type=int, default=None)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    cfg = default_cfg()
    for k, v in [("gk_tau", args.gk_tau), ("lam", args.lam),
                 ("s1_epochs", args.s1_epochs), ("s2_epochs", args.s2_epochs)]:
        if v is not None:
            cfg[k] = v
    if args.quick:
        cfg.update(n=6000, s1_epochs=12, s1_bs=64, s2_bs=128, s2_epochs=30, probe_epochs=25)

    print(f"device: {DEVICE}")
    if args.sanity:
        sanity(cfg, args.seeds); return

    sweeps = ["capacity", "rho", "nuisance", "alpha"] if args.sweep == "all" else [args.sweep]
    rows = []
    for sw in sweeps:
        print(f"\n{'#' * 90}\n# sweep: {sw}\n{'#' * 90}")
        # order jobs so d_emb varies fastest -> dataset and Stage-1 caches hit
        jobs = sorted([(dn, a, r, de, s) for (dn, a, r, de) in build_grid(sw, cfg)
                       for s in range(args.seeds)],
                      key=lambda j: (j[0], j[1], j[2], j[4], j[3]))
        for (d_n, alpha, rho, de, seed) in jobs:
            c = dict(cfg); c["d_emb"] = de
            t0 = time.time()
            r = run_condition(c, d_n, alpha, rho, seed)
            r["sweep"] = sw; rows.append(r)
            zn = ("n/a" if np.isnan(r["r2_zn_clip"])
                  else f"{r['r2_zn_clip']:.2f}/{r['r2_zn_supgk']:.2f}")
            tag = " [ANCHOR]" if is_anchor(d_n, alpha, rho) else ""
            print(f"[{sw}] d_n={d_n:4d} rho={rho:.2f} a={alpha:.2f} emb={de:4d} s={seed}"
                  f"  ({time.time()-t0:.0f}s){tag}\n"
                  f"   I_joint={r['I_joint']:.3f} ({r['frac_HY']:.0f}% H(Y)) "
                  f"acc_j={r['acc_j']:.3f} asym={r['asym']:.3f}\n"
                  f"   src%(joint)  CLIP={r['src_joint_clip']:6.1f}   "
                  f"SupGK={r['src_joint_supgk']:6.1f}   "
                  f"probe_acc {r['probe_acc_clip']:.2f}/{r['probe_acc_supgk']:.2f}\n"
                  f"   R2(z_c) {r['r2_zc_clip']:.2f}/{r['r2_zc_supgk']:.2f}  R2(z_n) {zn}  "
                  f"effrank {r['effrank_clip']:.0f}/{r['effrank_supgk']:.0f}")
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=2)
        summarize(rows, sw, cfg)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()