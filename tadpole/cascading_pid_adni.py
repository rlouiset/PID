"""
Cascading PID on ADNI/TADPOLE via R² decomposition.

Core idea:
  I(Y; (X1,X2)) / H(Y) ≈ R²_joint
  I(Y; X1) / H(Y)       ≈ R²_old
  I(Y; X2) / H(Y)       ≈ R²_new
  R / H(Y)               ≈ R²_redundancy  (from GK predictor)
  R_src / H(Y)           ≈ R²_source      (from source GK predictor)

Then:
  U_old = R²_old - R²_red
  U_new = R²_new - R²_red
  S     = R²_joint - R²_old - R²_new + R²_red
  R_mech = R²_red - R²_src
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import gc
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from copy import deepcopy
from collections import OrderedDict


# ============================================================
# 1. DATA
# ============================================================

def load_adni_data(tadpole_path, adnimerge_path):
    df_tadpole = pd.read_csv(tadpole_path, low_memory=False)
    df_tadpole = df_tadpole[df_tadpole["VISCODE"] == "bl"]
    df_tadpole = df_tadpole[df_tadpole["FLDSTRENG"].notna()]
    df_tadpole = df_tadpole[df_tadpole["FDG"].notna()]
    df_tadpole = df_tadpole[df_tadpole["AV45"].notna()]

    df_adni = pd.read_csv(adnimerge_path, low_memory=False)
    df_adni = df_adni[df_adni["VISCODE"] == "bl"]
    df_adni = df_adni[df_adni["ABETA_bl"].notna() & df_adni["TAU_bl"].notna() & df_adni["PTAU_bl"].notna()]
    df_adni = df_adni[df_adni["FDG"].notna() & df_adni["AV45"].notna()]

    df = pd.merge(df_tadpole, df_adni, on=["RID", "VISCODE"], how="inner", suffixes=("", "_adni"))
    df["PTGENDER"] = df["PTGENDER"].map({"Male": 1, "Female": 0})

    volumes = ["Ventricles_bl", "Hippocampus_bl", "Entorhinal_bl", "Fusiform_bl", "MidTemp_bl"]
    for col in ["ABETA_bl", "TAU_bl", "PTAU_bl", "ICV", "ADAS13"] + volumes:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in volumes:
        df[col + "_norm"] = df[col] / df["ICV"]
    return df


def define_modalities():
    vols = [f"{v}_norm" for v in
            ["Ventricles_bl", "Hippocampus_bl", "Entorhinal_bl", "Fusiform_bl", "MidTemp_bl"]]
    return OrderedDict([
        ("Demographics",   ["AGE", "PTEDUCAT", "PTGENDER", "APOE4"]),
        ("CSF_Amyloid",    ["ABETA_bl"]),
        ("CSF_Tau",        ["TAU_bl", "PTA_bl"]),
        ("Volumetric_MRI", vols),
        ("FDG_PET",        ["FDG"]),
    ])


def prepare_splits(df, modalities, target="ADAS13", test_size=0.3, val_size=0.25, seed=42):
    all_feats = sum(modalities.values(), [])
    df_m = df[all_feats + [target]].dropna().reset_index(drop=True)
    print(f"N = {len(df_m)}, target mean = {df_m[target].mean():.2f}, std = {df_m[target].std():.2f}")

    y = df_m[target].values.astype(np.float32)
    X = {name: df_m[feats].values.astype(np.float32) for name, feats in modalities.items()}

    idx = np.arange(len(y))
    i_tr, i_te, y_tr, y_te = train_test_split(idx, y, test_size=test_size, random_state=seed)
    i_va, i_te, y_va, y_te = train_test_split(i_te, y_te, test_size=val_size, random_state=seed)

    splits = {}
    for name, ii, yy in [("train", i_tr, y_tr), ("val", i_va, y_va), ("test", i_te, y_te)]:
        splits[name] = {"X": {k: v[ii] for k, v in X.items()}, "y": yy}
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


def xgb_r2(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost and return test R² (clipped to 0)."""
    m = make_xgb()
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    r2 = r2_score(y_test, m.predict(X_test))
    return max(0.0, r2)


def concat(X_dict, names):
    return np.concatenate([X_dict[n] for n in names], axis=1)


# ============================================================
# 3. SUPERVISED GACS-KORNER (for redundancy R²)
# ============================================================

class GKProjector(nn.Module):
    """Two projectors constrained to agree + shared prediction head."""
    def __init__(self, dim_a, dim_b, proj_dim=1024, hid=1024*2):
        super().__init__()
        self.proj_a = nn.Sequential(nn.Linear(dim_a, hid), nn.ReLU(), nn.Linear(hid, proj_dim))
        self.proj_b = nn.Sequential(nn.Linear(dim_b, hid), nn.ReLU(), nn.Linear(hid, proj_dim))
        self.head = nn.Sequential(nn.Linear(proj_dim, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, xa, xb):
        za = F.normalize(self.proj_a(xa), dim=-1)
        zb = F.normalize(self.proj_b(xb), dim=-1)
        # stop-gradient on embeddings for prediction head
        pa = self.head(za.detach()).squeeze(-1)
        pb = self.head(zb.detach()).squeeze(-1)
        pavg = self.head(((za.detach() + zb.detach()) / 2)).squeeze(-1)
        return za, zb, pa, pb, pavg


def sup_gk_loss(za, zb, y, tau=0.1, sigma_y=1, align_w=50.0):
    """Supervised GK contrastive + alignment loss."""
    K = torch.exp(-torch.cdist(za, zb).pow(2) / (2 * tau))
    S = torch.exp(-torch.cdist(y[:, None], y[:, None]).pow(2) / (2 * sigma_y**2))

    pos = ((K - 1)**2 * S).sum() / S.sum()
    neg = (K**2 * (1 - S)).sum() / (1 - S).sum()
    align = (za - zb).norm(dim=1).pow(2).mean()
    return pos + neg + align_w * align


def train_gk_and_get_r2(Xa_tr, Xb_tr, y_tr,
                         Xa_va, Xb_va, y_va,
                         Xa_te, Xb_te, y_te,
                         align_w=50.0, epochs=500, lr=1e-3,
                         bs=64, patience=50, device="cpu"):
    """
    Train supervised GK, return R² of the redundancy predictor on test.
    Returns both total redundancy R² and source redundancy R²:
      - R²_total: from the average-embedding predictor (captures all R)
      - R²_source: from per-modality predictors constrained to agree (source R only)
    """
    da, db = Xa_tr.shape[1], Xb_tr.shape[1]
    model = GKProjector(da, db).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    ta = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    xa_tr, xb_tr, yt = ta(Xa_tr), ta(Xb_tr), ta(y_tr)
    xa_va, xb_va, yv = ta(Xa_va), ta(Xb_va), ta(y_va)
    xa_te, xb_te     = ta(Xa_te), ta(Xb_te)

    loader = DataLoader(TensorDataset(xa_tr, xb_tr, yt), batch_size=bs, shuffle=True)

    best_vl, best_st, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xab, xbb, yb in loader:
            za, zb, pa, pb, pavg = model(xab, xbb)
            loss = (sup_gk_loss(za, zb, yb, align_w=align_w)
                    + F.mse_loss(pa, yb) + F.mse_loss(pb, yb) + F.mse_loss(pavg, yb))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            zav, zbv, _, _, _ = model(xa_va, xb_va)
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
        za_te, zb_te, pa_te, pb_te, pavg_te = model(xa_te, xb_te)

    pred_avg = pavg_te.cpu().numpy()
    pred_a   = pa_te.cpu().numpy()
    pred_b   = pb_te.cpu().numpy()

    # R²_redundancy: average embedding prediction
    r2_red = max(0.0, r2_score(y_te, pred_avg))

    # R²_source: worst of the two per-modality predictions
    # (source redundancy = info extractable as a common function of BOTH)
    r2_a = max(0.0, r2_score(y_te, pred_a))
    r2_b = max(0.0, r2_score(y_te, pred_b))
    r2_src = min(r2_a, r2_b)  # I_min principle on GK embeddings

    align_mse = float(np.mean((za_te.cpu().numpy() - zb_te.cpu().numpy())**2))

    return r2_red, r2_src, align_mse


# ============================================================
# 4. CASCADE
# ============================================================

def run_cascade(splits, modalities, device="cpu"):
    names = list(modalities.keys())
    base = "Demographics"
    rest = [m for m in names if m != base]

    y_tr, y_va, y_te = splits["train"]["y"], splits["val"]["y"], splits["test"]["y"]

    # ── Step 0 ──
    print(f"\n{'='*60}\nStep 0: {base}\n{'='*60}")
    r2_base = xgb_r2(
        splits["train"]["X"][base], y_tr,
        splits["val"]["X"][base],   y_va,
        splits["test"]["X"][base],  y_te,
    )
    print(f"  R² = {r2_base:.4f}  →  {r2_base:.1%} of H(Y) explained")

    results = [{
        "step": 0, "added": base, "accumulated": [base],
        "R2_joint": r2_base,
        "R": 0, "U_old": r2_base, "U_new": 0, "S": 0,
        "R_source": 0, "R_mech": 0,
        "unexplained": 1 - r2_base,
    }]

    accumulated = [base]

    for step in range(len(rest)):
        print(f"\n{'='*60}\nStep {step+1}: Greedy selection\n{'='*60}")
        candidates = [m for m in rest if m not in accumulated]
        if not candidates:
            break

        # ── Pick best ──
        best_r2, best_mod = -1, None
        for cand in candidates:
            trial = accumulated + [cand]
            r2 = xgb_r2(
                concat(splits["train"]["X"], trial), y_tr,
                concat(splits["val"]["X"], trial),   y_va,
                concat(splits["test"]["X"], trial),  y_te,
            )
            delta = r2 - results[-1]["R2_joint"]
            print(f"  {cand:<18s}: R² = {r2:.4f}  (Δ = {delta:+.4f})")
            if r2 > best_r2:
                best_r2, best_mod = r2, cand

        print(f"\n  --> Adding: {best_mod}")
        accumulated.append(best_mod)

        # ── Compute the 3 R² values ──
        r2_old = xgb_r2(
            concat(splits["train"]["X"], accumulated[:-1]), y_tr,
            concat(splits["val"]["X"],   accumulated[:-1]), y_va,
            concat(splits["test"]["X"],  accumulated[:-1]), y_te,
        )
        r2_new = xgb_r2(
            splits["train"]["X"][best_mod], y_tr,
            splits["val"]["X"][best_mod],   y_va,
            splits["test"]["X"][best_mod],  y_te,
        )
        r2_joint = xgb_r2(
            concat(splits["train"]["X"], accumulated), y_tr,
            concat(splits["val"]["X"],   accumulated), y_va,
            concat(splits["test"]["X"],  accumulated), y_te,
        )

        # ── Redundancy R² via supervised GK ──
        r2_red, r2_src, align_mse = train_gk_and_get_r2(
            concat(splits["train"]["X"], accumulated[:-1]),
            splits["train"]["X"][best_mod], y_tr,
            concat(splits["val"]["X"], accumulated[:-1]),
            splits["val"]["X"][best_mod], y_va,
            concat(splits["test"]["X"], accumulated[:-1]),
            splits["test"]["X"][best_mod], y_te,
            align_w=10.0, device=device,
        )

        # ── Solve the linear system ──
        # Clip R²_red to not exceed min(R²_old, R²_new) — PID constraint
        r2_red = min(r2_red, r2_old, r2_new)
        r2_src = min(r2_src, r2_red)

        U_old = r2_old - r2_red
        U_new = r2_new - r2_red
        R     = r2_red
        S     = r2_joint - r2_old - r2_new + r2_red
        R_src = r2_src
        R_mech = R - R_src

        # Clip negatives (finite-sample noise)
        U_old  = max(0, U_old)
        U_new  = max(0, U_new)
        S      = max(0, S)
        R_mech = max(0, R_mech)

        unexplained = max(0, 1 - r2_joint)

        print(f"\n  R²_old   = {r2_old:.4f}  ({', '.join(accumulated[:-1])})")
        print(f"  R²_new   = {r2_new:.4f}  ({best_mod})")
        print(f"  R²_joint = {r2_joint:.4f}")
        print(f"  R²_red   = {r2_red:.4f}  (GK redundancy)")
        print(f"  R²_src   = {r2_src:.4f}  (GK source)")
        print(f"  Align MSE = {align_mse:.6f}")
        print(f"\n  PID (as fraction of H(Y)):")
        print(f"    R       = {R:.4f}  (source: {R_src:.4f}, mech: {R_mech:.4f})")
        print(f"    U_old   = {U_old:.4f}")
        print(f"    U_new   = {U_new:.4f}")
        print(f"    S       = {S:.4f}")
        print(f"    Total   = {r2_joint:.4f}")
        print(f"    Unexp.  = {unexplained:.4f}")

        results.append({
            "step": step + 1, "added": best_mod,
            "accumulated": list(accumulated),
            "R2_joint": r2_joint, "R2_old": r2_old, "R2_new": r2_new,
            "R2_red": r2_red, "R2_src": r2_src,
            "R": R, "U_old": U_old, "U_new": U_new, "S": S,
            "R_source": R_src, "R_mech": R_mech,
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
    if n == 1: axes = [axes]

    C = {
        "R_source":    "#E8593C",
        "R_mech":      "#F2A623",
        "U_old":       "#3B8BD4",
        "U_new":       "#5DCAA5",
        "S":           "#7F77DD",
        "unexplained": "#D3D1C7",
    }

    for i, (ax, r) in enumerate(zip(axes, results)):
        if i == 0:
            vals = [r["U_old"], r["unexplained"]]
            labs = [f'Explained\n{r["U_old"]:.1%}', f'Unexplained\n{r["unexplained"]:.1%}']
            cols = [C["U_old"], C["unexplained"]]
        else:
            vals = [r["R_source"], r["R_mech"], r["U_old"], r["U_new"], r["S"], r["unexplained"]]
            labs = [f'R_src\n{r["R_source"]:.1%}', f'R_mech\n{r["R_mech"]:.1%}',
                    f'U_old\n{r["U_old"]:.1%}', f'U_new\n{r["U_new"]:.1%}',
                    f'S\n{r["S"]:.1%}', f'Unexp.\n{r["unexplained"]:.1%}']
            cols = [C["R_source"], C["R_mech"], C["U_old"], C["U_new"], C["S"], C["unexplained"]]

        # Drop tiny slices
        keep = [(v, l, c) for v, l, c in zip(vals, labs, cols) if v > 0.005]
        if keep:
            vals, labs, cols = zip(*keep)

        ax.pie(vals, labels=labs, colors=cols, startangle=90,
               textprops={"fontsize": 7},
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})

        title = f"Step {i}: +{r['added']}" if i > 0 else f"Step 0: {r['added']}"
        ax.set_title(f"{title}\nR²={r['R2_joint']:.3f}", fontsize=9, fontweight="bold")

    patches = [
        mpatches.Patch(color=C["R_source"],   label="Redundancy (source)"),
        mpatches.Patch(color=C["R_mech"],     label="Redundancy (mechanistic)"),
        mpatches.Patch(color=C["U_old"],      label="Unique (accumulated)"),
        mpatches.Patch(color=C["U_new"],      label="Unique (new modality)"),
        mpatches.Patch(color=C["S"],          label="Synergy"),
        mpatches.Patch(color=C["unexplained"],label="Unexplained"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close()


def print_summary(results):
    print(f"\n{'='*95}")
    print(f"CASCADING PID SUMMARY (all values as fraction of H(Y))")
    print(f"{'='*95}")
    print(f"{'Step':<5} {'Added':<18} {'R²':>6} {'R':>6} {'R_src':>6} {'R_mch':>6} "
          f"{'U_old':>6} {'U_new':>6} {'S':>6} {'Unexp':>6}")
    print("-" * 95)
    for r in results:
        print(f"{r['step']:<5} {r['added']:<18} {r['R2_joint']:>6.3f} "
              f"{r['R']:>6.3f} {r['R_source']:>6.3f} {r['R_mech']:>6.3f} "
              f"{r['U_old']:>6.3f} {r['U_new']:>6.3f} {r['S']:>6.3f} "
              f"{r['unexplained']:>6.3f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tadpole-path", type=str, required=True)
    parser.add_argument("--adnimerge-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="./cascading_pid_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_adni_data(args.tadpole_path, args.adnimerge_path)
    modalities = define_modalities()
    for name, feats in modalities.items():
        print(f"  {name}: {feats}")

    splits = prepare_splits(df, modalities)
    results = run_cascade(splits, modalities, device=args.device)
    print_summary(results)

    with open(os.path.join(args.output_dir, "cascade_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: list(x) if isinstance(x, np.ndarray) else x)

    plot_cascade(results, save_path=os.path.join(args.output_dir, "cascading_pid.pdf"))
    print("Done!")