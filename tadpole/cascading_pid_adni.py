"""
Cascading PID on ADNI/TADPOLE via variance-fraction decomposition.

All PID quantities are fractions of Var(Y) and sum to 1:
  R + U_old + U_new + S + Unexplained = 1

Pointwise information (per sample):
  i(x_i; y_i) = [(y_i - ȳ)² - (y_i - f(x_i))²] / Var(Y)
  → positive = model helps, negative = model misinforms
  → E[i] = R²

Redundancy decomposition:
  R_total  = E[min(i_old, i_new) * 1{both > 0}]   (sign-filtered I_min)
  R_source = R²_GK  (from Gács-Körner worst-head predictor)
  R_mech   = R_total - R_source

PID:
  U_old = R²_old - R_total
  U_new = R²_new - R_total
  S     = R²_joint - R²_old - R²_new + R_total
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from copy import deepcopy
from collections import OrderedDict


# ============================================================
# 0. POINTWISE PID
# ============================================================

def pointwise_info(y, y_pred, var_y, y_bar):
    """
    Per-sample information:
      i(x_i; y_i) = [(y_i - ȳ)² - (y_i - ŷ_i)²] / Var(Y)
    """
    return ((y - y_bar) ** 2 - (y - y_pred) ** 2) / var_y


def compute_pid(y, y_pred_old, y_pred_new, y_pred_joint):
    """
    Full pointwise PID → averaged to global values.
    All values are fractions of Var(Y).
    """
    var_y = np.var(y)
    y_bar = np.mean(y)

    i_old = pointwise_info(y, y_pred_old, var_y, y_bar)
    i_new = pointwise_info(y, y_pred_new, var_y, y_bar)
    i_joint = pointwise_info(y, y_pred_joint, var_y, y_bar)

    # Sign-filtered I_min redundancy
    both_pos = (i_old > 0) & (i_new > 0)
    r = np.where(both_pos, np.minimum(i_old, i_new), 0.0)

    u_old = i_old - r
    u_new = i_new - r
    s = i_joint - i_old - i_new + r

    R = float(np.mean(r))
    U_old = float(np.mean(u_old))
    U_new = float(np.mean(u_new))
    S = float(np.mean(s))

    # R² values for reference
    mse_old = mean_squared_error(y, y_pred_old)
    mse_new = mean_squared_error(y, y_pred_new)
    mse_joint = mean_squared_error(y, y_pred_joint)
    r2_old = max(0.0, 1 - mse_old / var_y)
    r2_new = max(0.0, 1 - mse_new / var_y)
    r2_joint = max(0.0, 1 - mse_joint / var_y)

    return {
        "R": R, "U_old": U_old, "U_new": U_new, "S": S,
        "R2_old": r2_old, "R2_new": r2_new, "R2_joint": r2_joint,
        "pointwise": {
            "i_old": i_old, "i_new": i_new, "i_joint": i_joint, "r": r,
            "u_old": u_old, "u_new": u_new, "s": s,
        },
    }


# ============================================================
# 1. DATA
# ============================================================

def load_and_merge(tadpole_path, adnimerge_path):
    """Load a TADPOLE CSV and merge with ADNIMERGE."""
    df_tadpole = pd.read_csv(tadpole_path, low_memory=False)
    df_tadpole = df_tadpole[df_tadpole["VISCODE"] == "bl"]
    df_tadpole = df_tadpole[df_tadpole["FLDSTRENG"].notna()]
    df_tadpole = df_tadpole[df_tadpole["FDG"].notna()]
    df_tadpole = df_tadpole[df_tadpole["AV45"].notna()]

    df_adni = pd.read_csv(adnimerge_path, low_memory=False)
    df_adni = df_adni[df_adni["VISCODE"] == "bl"]
    df_adni = df_adni[
        df_adni["ABETA_bl"].notna()
        & df_adni["TAU_bl"].notna()
        & df_adni["PTAU_bl"].notna()
    ]
    df_adni = df_adni[df_adni["FDG"].notna() & df_adni["AV45"].notna()]

    df = pd.merge(
        df_tadpole, df_adni,
        on=["RID", "VISCODE"], how="inner", suffixes=("", "_adni"),
    )
    df["PTGENDER"] = df["PTGENDER"].map({"Male": 1, "Female": 0})

    volumes = [
        "Ventricles_bl", "Hippocampus_bl", "Entorhinal_bl",
        "Fusiform_bl", "MidTemp_bl",
    ]
    for col in ["ABETA_bl", "TAU_bl", "PTAU_bl", "ICV", "ADAS13"] + volumes:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in volumes:
        df[col + "_norm"] = df[col] / df["ICV"]
    return df


def define_modalities():
    vols = [
        f"{v}_norm"
        for v in [
            "Ventricles_bl", "Hippocampus_bl", "Entorhinal_bl",
            "Fusiform_bl", "MidTemp_bl",
        ]
    ]
    return OrderedDict([
        ("Demographics", ["AGE", "PTEDUCAT", "PTGENDER", "APOE4"]),
        ("CSF_Amyloid", ["ABETA_bl"]),
        ("CSF_Tau", ["TAU_bl", "PTAU_bl"]),
        ("Volumetric_MRI", vols),
        ("FDG_PET", ["FDG"]),
    ])


def prepare_data(df, modalities, target="ADAS13",
                 test_size=0.25, val_size=0.2, seed=42):
    """
    Split single dataframe into train/val/test.
    Val is used only for early stopping / hyperparameter tuning.
    """
    all_feats = sum(modalities.values(), [])
    cols = all_feats + [target]

    df_m = df[cols].dropna().reset_index(drop=True)
    y = df_m[target].values.astype(np.float32)
    X = {n: df_m[f].values.astype(np.float32) for n, f in modalities.items()}

    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=test_size, random_state=seed)
    i_tr, i_va = train_test_split(i_tr, test_size=val_size, random_state=seed)

    splits = {}
    for name, ii in [("train", i_tr), ("val", i_va), ("test", i_te)]:
        splits[name] = {"X": {k: v[ii] for k, v in X.items()}, "y": y[ii]}

    print(f"N = {len(df_m)}, Train={len(i_tr)}, Val={len(i_va)}, Test={len(i_te)}")
    print(f"Target: mean={y[i_te].mean():.2f}, std={y[i_te].std():.2f}, "
          f"var={np.var(y[i_te]):.2f}")

    return splits


# ============================================================
# 2. XGBOOST
# ============================================================

def make_xgb():
    return XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0, min_child_weight=3,
        random_state=42, verbosity=0,
    )


def fit_and_predict(X_train, y_train, X_val, y_val, X_test):
    """
    Train XGBoost with val for early stopping, return predictions on test.
    """
    m = make_xgb()
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return m.predict(X_test)


def concat(X_dict, names):
    return np.concatenate([X_dict[n] for n in names], axis=1)


# ============================================================
# 3. GÁCS-KÖRNER (for source redundancy)
# ============================================================

class GKProjector(nn.Module):
    """Two projectors constrained to agree + shared prediction head."""

    def __init__(self, dim_a, dim_b, proj_dim=1024, hid=2048):
        super().__init__()
        self.proj_a = nn.Sequential(
            nn.Linear(dim_a, hid), nn.ReLU(), nn.Linear(hid, proj_dim),
        )
        self.proj_b = nn.Sequential(
            nn.Linear(dim_b, hid), nn.ReLU(), nn.Linear(hid, proj_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(proj_dim, hid), nn.ReLU(), nn.Linear(hid, 1),
        )

    def forward(self, xa, xb):
        za = F.normalize(self.proj_a(xa), dim=-1)
        zb = F.normalize(self.proj_b(xb), dim=-1)
        pa = self.head(za.detach()).squeeze(-1)
        pb = self.head(zb.detach()).squeeze(-1)
        return za, zb, pa, pb


def sup_gk_loss(za, zb, y, tau=0.1, sigma_y=1.0, align_w=50.0):
    """Supervised GK contrastive + alignment loss."""
    K = torch.exp(-torch.cdist(za, zb).pow(2) / (2 * tau))
    S = torch.exp(-torch.cdist(y[:, None], y[:, None]).pow(2) / (2 * sigma_y ** 2))
    pos = ((K - 1) ** 2 * S).sum() / S.sum()
    neg = (K ** 2 * (1 - S)).sum() / (1 - S).sum()
    align = (za - zb).norm(dim=1).pow(2).mean()
    return pos + neg + align_w * align


def train_gk(Xa_tr, Xb_tr, y_tr, Xa_va, Xb_va, y_va,
             Xa_te, Xb_te, y_te,
             align_w=50.0, epochs=500, lr=1e-3, bs=64,
             patience=50, device="cpu"):
    """
    Train supervised GK projector.
    Returns R²_source on test = R² of worst per-modality head.
    """
    da, db = Xa_tr.shape[1], Xb_tr.shape[1]
    model = GKProjector(da, db).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    ta = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    xa_tr, xb_tr, yt = ta(Xa_tr), ta(Xb_tr), ta(y_tr)
    xa_va, xb_va, yv = ta(Xa_va), ta(Xb_va), ta(y_va)
    xa_te, xb_te = ta(Xa_te), ta(Xb_te)

    loader = DataLoader(
        TensorDataset(xa_tr, xb_tr, yt), batch_size=bs, shuffle=True,
    )

    best_vl, best_st, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xab, xbb, yb in loader:
            za, zb, pa, pb = model(xab, xbb)
            loss = (
                sup_gk_loss(za, zb, yb, align_w=align_w)
                + F.mse_loss(pa, yb)
                + F.mse_loss(pb, yb)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            zav, zbv, _, _ = model(xa_va, xb_va)
            vl = sup_gk_loss(zav, zbv, yv, align_w=align_w).item()
        if vl < best_vl:
            best_vl, best_st, wait = vl, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_st)
    model.eval()

    with torch.no_grad():
        _, _, pa_te, pb_te = model(xa_te, xb_te)

    pred_a = pa_te.cpu().numpy()
    pred_b = pb_te.cpu().numpy()

    # Source redundancy = R² of the WORST per-modality head (I_min on GK)
    var_y = np.var(y_te)
    mse_a = mean_squared_error(y_te, pred_a)
    mse_b = mean_squared_error(y_te, pred_b)
    r2_a = max(0.0, 1 - mse_a / var_y)
    r2_b = max(0.0, 1 - mse_b / var_y)
    r2_source = min(r2_a, r2_b)

    return r2_source


# ============================================================
# 4. CASCADE
# ============================================================

def run_cascade(splits, modalities, device="cpu"):
    names = list(modalities.keys())
    base = "Demographics"
    rest = [m for m in names if m != base]

    y_tr = splits["train"]["y"]
    y_va = splits["val"]["y"]
    y_te = splits["test"]["y"]
    var_y = np.var(y_te)

    # ── Step 0: base modality only ──
    print(f"\n{'=' * 60}\nStep 0: {base}\n{'=' * 60}")
    pred_base = fit_and_predict(
        splits["train"]["X"][base], y_tr,
        splits["val"]["X"][base], y_va,
        splits["test"]["X"][base],
    )
    r2_base = max(0.0, r2_score(y_te, pred_base))
    print(f"  R² = {r2_base:.4f}")

    results = [{
        "step": 0, "added": base, "accumulated": [base],
        "R2_joint": r2_base,
        "R": 0, "U_old": r2_base, "U_new": 0, "S": 0,
        "R_source": 0, "R_mech": 0,
        "unexplained": 1 - r2_base,
    }]

    accumulated = [base]

    for step in range(len(rest)):
        print(f"\n{'=' * 60}\nStep {step + 1}: Greedy selection\n{'=' * 60}")
        candidates = [m for m in rest if m not in accumulated]
        if not candidates:
            break

        # ── Pick the modality that maximizes joint R² ──
        best_r2, best_mod = -1, None
        for cand in candidates:
            trial = accumulated + [cand]
            pred = fit_and_predict(
                concat(splits["train"]["X"], trial), y_tr,
                concat(splits["val"]["X"], trial), y_va,
                concat(splits["test"]["X"], trial),
            )
            r2 = max(0.0, r2_score(y_te, pred))
            delta = r2 - results[-1]["R2_joint"]
            print(f"  {cand:<18s}: R² = {r2:.4f}  (Δ = {delta:+.4f})")
            if r2 > best_r2:
                best_r2, best_mod = r2, cand

        print(f"\n  --> Adding: {best_mod}")
        accumulated.append(best_mod)

        # ── Get predictions for old, new, joint ──
        pred_old = fit_and_predict(
            concat(splits["train"]["X"], accumulated[:-1]), y_tr,
            concat(splits["val"]["X"], accumulated[:-1]), y_va,
            concat(splits["test"]["X"], accumulated[:-1]),
        )
        pred_new = fit_and_predict(
            splits["train"]["X"][best_mod], y_tr,
            splits["val"]["X"][best_mod], y_va,
            splits["test"]["X"][best_mod],
        )
        pred_joint = fit_and_predict(
            concat(splits["train"]["X"], accumulated), y_tr,
            concat(splits["val"]["X"], accumulated), y_va,
            concat(splits["test"]["X"], accumulated),
        )

        # ── Pointwise PID (R_total via sign-filtered I_min) ──
        pid = compute_pid(y_te, pred_old, pred_new, pred_joint)

        # ── Source redundancy via GK ──
        r2_source = train_gk(
            concat(splits["train"]["X"], accumulated[:-1]),
            splits["train"]["X"][best_mod], y_tr,
            concat(splits["val"]["X"], accumulated[:-1]),
            splits["val"]["X"][best_mod], y_va,
            concat(splits["test"]["X"], accumulated[:-1]),
            splits["test"]["X"][best_mod], y_te,
            align_w=10.0, device=device,
        )

        # ── Clip and assemble ──
        R_total = pid["R"]
        R_source = min(r2_source, R_total)  # source can't exceed total
        R_mech = max(0.0, R_total - R_source)
        U_old = pid["U_old"]
        U_new = pid["U_new"]
        S = pid["S"]
        r2_joint = pid["R2_joint"]
        unexplained = max(0.0, 1 - r2_joint)

        print(f"\n  R²_old   = {pid['R2_old']:.4f}  ({', '.join(accumulated[:-1])})")
        print(f"  R²_new   = {pid['R2_new']:.4f}  ({best_mod})")
        print(f"  R²_joint = {r2_joint:.4f}")
        print(f"\n  PID (fraction of Var(Y)):")
        print(f"    R_total  = {R_total:.4f}  "
              f"(source: {R_source:.4f}, mech: {R_mech:.4f})")
        print(f"    U_old    = {U_old:.4f}")
        print(f"    U_new    = {U_new:.4f}")
        print(f"    S        = {S:.4f}")
        print(f"    Unexp.   = {unexplained:.4f}")
        print(f"    Sum      = {R_total + U_old + U_new + S + unexplained:.4f}")

        results.append({
            "step": step + 1,
            "added": best_mod,
            "accumulated": list(accumulated),
            "R2_joint": r2_joint,
            "R2_old": pid["R2_old"],
            "R2_new": pid["R2_new"],
            "R": R_total,
            "U_old": U_old,
            "U_new": U_new,
            "S": S,
            "R_source": R_source,
            "R_mech": R_mech,
            "unexplained": unexplained,
        })

    return results


# ============================================================
# 5. VISUALIZATION
# ============================================================

def plot_cascade(results, save_path="cascading_pid.pdf"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    if n == 1:
        axes = [axes]

    C = {
        "R_source": "#E8593C",
        "R_mech": "#F2A623",
        "U_old": "#3B8BD4",
        "U_new": "#5DCAA5",
        "S": "#7F77DD",
        "unexplained": "#D3D1C7",
    }

    for i, (ax, r) in enumerate(zip(axes, results)):
        if i == 0:
            vals = [r["U_old"], r["unexplained"]]
            labs = [
                f'Explained\n{r["U_old"]:.1%}',
                f'Unexplained\n{r["unexplained"]:.1%}',
            ]
            cols = [C["U_old"], C["unexplained"]]
        else:
            vals = [
                r["R_source"], r["R_mech"],
                r["U_old"], r["U_new"],
                r["S"], r["unexplained"],
            ]
            labs = [
                f'R_src\n{r["R_source"]:.1%}',
                f'R_mech\n{r["R_mech"]:.1%}',
                f'U_old\n{r["U_old"]:.1%}',
                f'U_new\n{r["U_new"]:.1%}',
                f'S\n{r["S"]:.1%}',
                f'Unexp.\n{r["unexplained"]:.1%}',
            ]
            cols = [
                C["R_source"], C["R_mech"],
                C["U_old"], C["U_new"],
                C["S"], C["unexplained"],
            ]

        # Drop tiny slices
        keep = [(v, l, c) for v, l, c in zip(vals, labs, cols) if v > 0.005]
        if keep:
            vals, labs, cols = zip(*keep)

        ax.pie(
            vals, labels=labs, colors=cols, startangle=90,
            textprops={"fontsize": 7},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        title = (
            f"Step {i}: +{r['added']}" if i > 0 else f"Step 0: {r['added']}"
        )
        ax.set_title(
            f"{title}\nR²={r['R2_joint']:.3f}", fontsize=9, fontweight="bold",
        )

    patches = [
        mpatches.Patch(color=C["R_source"], label="Redundancy (source)"),
        mpatches.Patch(color=C["R_mech"], label="Redundancy (mechanistic)"),
        mpatches.Patch(color=C["U_old"], label="Unique (accumulated)"),
        mpatches.Patch(color=C["U_new"], label="Unique (new modality)"),
        mpatches.Patch(color=C["S"], label="Synergy"),
        mpatches.Patch(color=C["unexplained"], label="Unexplained"),
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=3, fontsize=8,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close()


def print_summary(results):
    print(f"\n{'=' * 100}")
    print("CASCADING PID SUMMARY (all values as fraction of Var(Y))")
    print(f"{'=' * 100}")
    print(
        f"{'Step':<5} {'Added':<18} {'R²':>6} {'R':>6} {'R_src':>6} "
        f"{'R_mch':>6} {'U_old':>6} {'U_new':>6} {'S':>6} {'Unexp':>6}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['step']:<5} {r['added']:<18} {r['R2_joint']:>6.3f} "
            f"{r['R']:>6.3f} {r['R_source']:>6.3f} {r['R_mech']:>6.3f} "
            f"{r['U_old']:>6.3f} {r['U_new']:>6.3f} {r['S']:>6.3f} "
            f"{r['unexplained']:>6.3f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tadpole-path", type=str, required=True,
                        help="Path to TADPOLE CSV")
    parser.add_argument("--adnimerge-path", type=str, required=True,
                        help="Path to ADNIMERGE CSV")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str,
                        default="./cascading_pid_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    df = load_and_merge(args.tadpole_path, args.adnimerge_path)
    print(f"  {len(df)} subjects after merge")

    modalities = define_modalities()
    for name, feats in modalities.items():
        print(f"  {name}: {feats}")

    splits = prepare_data(df, modalities)
    results = run_cascade(splits, modalities, device=args.device)
    print_summary(results)

    # Save results
    results_serializable = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "pointwise"}
        r_copy = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in r_copy.items()
        }
        if isinstance(r_copy.get("accumulated"), list):
            r_copy["accumulated"] = list(r_copy["accumulated"])
        results_serializable.append(r_copy)

    with open(os.path.join(args.output_dir, "cascade_results.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)

    plot_cascade(results, save_path=os.path.join(args.output_dir, "cascading_pid.pdf"))
    print("Done!")