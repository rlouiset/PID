"""
Cascading PID on ADNI/TADPOLE — Predicting future ADAS13 slope.

Target: annual rate of ADAS13 change (points/year), estimated via
XGBoost per-patient regression of ADAS13 ~ time, using all available
longitudinal visits. Only patients with ≥24 months follow-up and
≥3 ADAS13 measurements are included.

Features: baseline (VISCODE == "bl") modalities only, including a
"Cognitive_Baseline" modality with ADAS13, MMSE, CDRSB, FAQ, RAVLT.
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
# 0. INFORMATION-THEORETIC HELPERS
# ============================================================

def H_Y(var_y):
    return 0.5 * np.log(2 * np.pi * np.e * var_y)

def MI_from_r2(r2):
    r2 = np.clip(r2, 0.0, 1.0 - 1e-12)
    return -0.5 * np.log(1 - r2)

def nats_to_r2(I_nats):
    return 1.0 - np.exp(-2.0 * max(0.0, I_nats))

def pid_nats_to_r2(R_nats, r2_old, r2_new, r2_joint, R_source_nats=None):
    R_nats_c   = max(0.0, R_nats)
    r2_R       = 1.0 - np.exp(-2.0 * R_nats_c)
    r2_U_old   = r2_old - r2_R
    r2_U_new   = r2_new - r2_R
    r2_S       = r2_joint - r2_R - r2_U_old - r2_U_new

    raw_sum = r2_R + r2_U_old + r2_U_new + r2_S
    scale   = (r2_joint / raw_sum) if abs(raw_sum) > 1e-12 else 1.0
    r2_R     *= scale
    r2_U_old *= scale
    r2_U_new *= scale
    r2_S     *= scale

    R_src = min(max(0.0, R_source_nats if R_source_nats is not None else 0.0), R_nats_c)
    r2_R_source = (1.0 - np.exp(-2.0 * R_src)) * scale
    r2_R_mech   = (np.exp(-2.0 * R_src) - np.exp(-2.0 * R_nats_c)) * scale

    return {
        "r2_R": r2_R, "r2_U_old": r2_U_old, "r2_U_new": r2_U_new, "r2_S": r2_S,
        "r2_R_source": r2_R_source, "r2_R_mech": r2_R_mech,
        "r2_unexplained": 1.0 - r2_joint,
    }

def pointwise_info(y, y_pred, var_y, y_bar, mse):
    const = 0.5 * np.log(var_y / mse)
    return const + (y - y_bar) ** 2 / (2 * var_y) - (y - y_pred) ** 2 / (2 * mse)

def compute_pid(y, y_pred_old, y_pred_new, y_pred_joint):
    var_y = np.var(y)
    y_bar = np.mean(y)
    mse_old = mean_squared_error(y, y_pred_old)
    mse_new = mean_squared_error(y, y_pred_new)
    mse_joint = mean_squared_error(y, y_pred_joint)

    i_old = pointwise_info(y, y_pred_old, var_y, y_bar, mse_old)
    i_new = pointwise_info(y, y_pred_new, var_y, y_bar, mse_new)
    i_joint = pointwise_info(y, y_pred_joint, var_y, y_bar, mse_joint)

    both_pos = (i_old > 0) & (i_new > 0)
    r = np.where(both_pos, np.minimum(i_old, i_new), 0.0)

    I_old = float(np.mean(i_old))
    I_new = float(np.mean(i_new))
    I_joint = float(np.mean(i_joint))
    R = float(np.mean(r))

    I_old = min(I_old, I_joint)
    I_new = min(I_new, I_joint)
    R = min(R, I_old, I_new, I_joint)

    U_old = I_old - R
    U_new = I_new - R
    S = I_joint - I_old - I_new + R

    r2_old = max(0.0, 1 - mse_old / var_y)
    r2_new = max(0.0, 1 - mse_new / var_y)
    r2_joint = max(0.0, 1 - mse_joint / var_y)

    return {
        "R": R, "U_old": U_old, "U_new": U_new, "S": S,
        "I_old": I_old, "I_new": I_new, "I_joint": I_joint,
        "R2_old": r2_old, "R2_new": r2_new, "R2_joint": r2_joint,
    }


# ============================================================
# 1. DATA — LONGITUDINAL SLOPE TARGET
# ============================================================

def compute_adas13_slopes(df_adni, min_months=24):
    """
    For each patient, compute cognitive slope as the average of
    (ADAS13_t - ADAS13_bl) / (t_months / 12) across ALL visits
    at t >= min_months.

    Units: ADAS13 points per year.
    """
    df_adni = df_adni.copy()
    df_adni["ADAS13"] = pd.to_numeric(df_adni["ADAS13"], errors="coerce")
    df_adni["Month"] = pd.to_numeric(df_adni["Month"], errors="coerce")

    slopes = []
    for rid, grp in df_adni.groupby("RID"):
        bl_row = grp[(grp["VISCODE"] == "bl") & grp["ADAS13"].notna()]
        if len(bl_row) == 0:
            continue
        adas_bl = float(bl_row["ADAS13"].iloc[0])

        future = grp[(grp["Month"] >= min_months) & grp["ADAS13"].notna()]
        if len(future) == 0:
            continue

        terms = []
        for _, row in future.iterrows():
            t_years = row["Month"] / 12.0
            delta = (row["ADAS13"] - adas_bl) / t_years
            terms.append(delta)

        slopes.append({
            "RID": rid,
            "adas13_slope": float(np.mean(terms)),
            "adas13_baseline": adas_bl,
            "n_future_visits": len(terms),
            "max_followup_years": float(future["Month"].max() / 12.0),
        })

    return pd.DataFrame(slopes)


def load_data_slope(tadpole_path, adnimerge_path, min_months=24):
    """
    Load baseline features from TADPOLE + ADNIMERGE, and compute
    the ADAS13 slope as target from all future visits >= min_months.
    """
    # --- Baseline features from TADPOLE ---
    df_tadpole = pd.read_csv(tadpole_path, low_memory=False)
    df_tadpole = df_tadpole[df_tadpole["VISCODE"] == "bl"]
    df_tadpole = df_tadpole[df_tadpole["FLDSTRENG"].notna()]
    df_tadpole = df_tadpole[df_tadpole["FDG"].notna()]
    df_tadpole = df_tadpole[df_tadpole["AV45"].notna()]

    # --- All visits from ADNIMERGE (for slope) ---
    df_adni_all = pd.read_csv(adnimerge_path, low_memory=False)
    slopes_df = compute_adas13_slopes(df_adni_all, min_months=min_months)
    print(f"  Patients with valid ADAS13 slope (visits >= {min_months}mo): {len(slopes_df)}")
    print(f"  Slope (pts/yr): mean={slopes_df['adas13_slope'].mean():.2f}, "
          f"std={slopes_df['adas13_slope'].std():.2f}, "
          f"median={slopes_df['adas13_slope'].median():.2f}")
    print(f"  Future visits per patient: "
          f"mean={slopes_df['n_future_visits'].mean():.1f}, "
          f"max follow-up: {slopes_df['max_followup_years'].max():.1f} yr")

    # --- Baseline row from ADNIMERGE (for CSF + cognitive scores) ---
    df_adni_bl = df_adni_all[df_adni_all["VISCODE"] == "bl"].copy()
    df_adni_bl = df_adni_bl[
        df_adni_bl["ABETA_bl"].notna()
        & df_adni_bl["TAU_bl"].notna()
        & df_adni_bl["PTAU_bl"].notna()
    ]

    # --- Merge ---
    df = pd.merge(
        df_tadpole, df_adni_bl,
        on=["RID", "VISCODE"], how="inner", suffixes=("", "_adni"),
    )
    df = pd.merge(df, slopes_df[["RID", "adas13_slope"]], on="RID", how="inner")

    df["PTGENDER"] = df["PTGENDER"].map({"Male": 1, "Female": 0})

    volumes = [
        "Ventricles_bl", "Hippocampus_bl", "Entorhinal_bl",
        "Fusiform_bl", "MidTemp_bl",
    ]
    numeric_cols = (
        ["ABETA_bl", "TAU_bl", "PTAU_bl", "ICV", "FDG"]
        + volumes
        + ["ADAS13", "MMSE", "CDRSB", "FAQ",
           "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting",
           "LDELTOTAL", "TRABSCOR"]
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in volumes:
        df[col + "_norm"] = df[col] / df["ICV"]

    print(f"  Final merged dataset: {len(df)} subjects")

    # Verify no duplicate RIDs
    assert df["RID"].is_unique, "Duplicate RIDs found after merge!"

    return df


def define_modalities():
    vols = [
        f"{v}_norm" for v in [
            "Ventricles_bl", "Hippocampus_bl", "Entorhinal_bl",
            "Fusiform_bl", "MidTemp_bl",
        ]
    ]
    return OrderedDict([
        ("Demographics", ["AGE", "PTEDUCAT", "PTGENDER", "APOE4"]),
        ("Cognitive_Baseline", [
            "ADAS13", "MMSE", "CDRSB", "FAQ",
            "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting",
            "LDELTOTAL", "TRABSCOR",
        ]),
        ("CSF_Amyloid", ["ABETA_bl"]),
        ("CSF_Tau", ["TAU_bl", "PTAU_bl"]),
        ("Volumetric_MRI", vols),
        ("FDG_PET", ["FDG"]),
    ])


def prepare_data(df, modalities, target="adas13_slope",
                 test_size=0.25, val_size=0.2, seed=42):
    all_feats = sum(modalities.values(), [])
    cols = all_feats + [target]

    df_m = df[cols].dropna().reset_index(drop=True)
    y = df_m[target].values.astype(np.float32)
    X = {n: df_m[f].values.astype(np.float32) for n, f in modalities.items()}

    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=test_size, random_state=seed)
    i_tr, i_va = train_test_split(i_tr, test_size=val_size, random_state=seed)

    # Explicit leakage check
    assert len(set(i_tr) & set(i_te)) == 0, "Train/test overlap!"
    assert len(set(i_tr) & set(i_va)) == 0, "Train/val overlap!"
    assert len(set(i_va) & set(i_te)) == 0, "Val/test overlap!"

    splits = {}
    for name, ii in [("train", i_tr), ("val", i_va), ("test", i_te)]:
        splits[name] = {"X": {k: v[ii] for k, v in X.items()}, "y": y[ii]}

    print(f"N = {len(df_m)}, Train={len(i_tr)}, Val={len(i_va)}, Test={len(i_te)}")
    print(f"Target (ADAS13 slope, pts/yr): mean={y[i_te].mean():.2f}, "
          f"std={y[i_te].std():.2f}, var={np.var(y[i_te]):.2f}")

    return splits


# ============================================================
# 2. XGBOOST WITH GRID SEARCH
# ============================================================

XGB_PARAM_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 400],
    "min_child_weight": [1, 3, 5],
}

_best_xgb_params = None  # cached after first grid search


def grid_search_xgb(X_train, y_train, X_val, y_val):
    """
    Grid search over XGB_PARAM_GRID, scored on val set via R².
    Returns best params dict.
    """
    from itertools import product as iterproduct

    keys = list(XGB_PARAM_GRID.keys())
    values = list(XGB_PARAM_GRID.values())

    best_r2, best_params = -np.inf, None
    n_combos = 1
    for v in values:
        n_combos *= len(v)
    print(f"    Grid search: {n_combos} combinations...")

    for combo in iterproduct(*values):
        params = dict(zip(keys, combo))
        m = XGBRegressor(
            **params,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0,
            random_state=42, verbosity=0,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = m.predict(X_val)
        r2 = r2_score(y_val, pred)
        if r2 > best_r2:
            best_r2, best_params = r2, params

    print(f"    Best val R² = {best_r2:.4f},  params = {best_params}")
    return best_params


def fit_and_predict(X_train, y_train, X_val, y_val, X_test):
    """
    Train XGBoost with best hyperparameters (grid searched once,
    then cached for subsequent calls).
    """
    print("  Running XGBoost grid search on val set...")
    _best_xgb_params = grid_search_xgb(X_train, y_train, X_val, y_val)

    m = XGBRegressor(
        **_best_xgb_params,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0,
        random_state=42, verbosity=0,
    )
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return m.predict(X_test)


def concat(X_dict, names):
    return np.concatenate([X_dict[n] for n in names], axis=1)


# ============================================================
# 3. GACS-KORNER (for source redundancy)
# ============================================================

class GKProjector(nn.Module):
    def __init__(self, dim_a, dim_b, proj_dim=32, hid=1024):
        super().__init__()
        self.proj_a = nn.Sequential(
            nn.Linear(dim_a, hid * 2), nn.ReLU(), nn.Linear(hid * 2, proj_dim),
        )
        self.proj_b = nn.Sequential(
            nn.Linear(dim_b, hid * 2), nn.ReLU(), nn.Linear(hid * 2, proj_dim),
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

def sup_gk_loss(za, zb, y, tau=0.1, sigma_y=1.0, align_w=10.0):
    K = torch.exp(-torch.cdist(za, zb).pow(2) / (2 * tau))
    S = torch.exp(-torch.cdist(y[:, None], y[:, None]).pow(2) / (2 * sigma_y ** 2))
    pos = ((K - 1) ** 2 * S).sum() / S.sum()
    neg = (K ** 2 * (1 - S)).sum() / (1 - S).sum()
    align = (za - zb).norm(dim=1).pow(2).mean()
    return pos + neg + align_w * align

def train_gk(Xa_tr, Xb_tr, y_tr, Xa_va, Xb_va, y_va,
             Xa_te, Xb_te, y_te,
             align_w=10.0, epochs=100, lr=1e-3, bs=1024,
             patience=10, device="cpu"):
    da, db = Xa_tr.shape[1], Xb_tr.shape[1]
    model = GKProjector(da, db).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    ta = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    xa_tr, xb_tr, yt = ta(Xa_tr), ta(Xb_tr), ta(y_tr)
    xa_va, xb_va, yv = ta(Xa_va), ta(Xb_va), ta(y_va)
    xa_te, xb_te = ta(Xa_te), ta(Xb_te)

    loader = DataLoader(TensorDataset(xa_tr, xb_tr, yt), batch_size=bs, shuffle=True)

    best_vl, best_st, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xab, xbb, yb in loader:
            za, zb, pa, pb = model(xab, xbb)
            loss = (sup_gk_loss(za, zb, yb, align_w=align_w)
                    + F.mse_loss(pa, yb) + F.mse_loss(pb, yb))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            zav, zbv, _, _ = model(xa_va, xb_va)
            vl = sup_gk_loss(zav, zbv, yv, align_w=align_w).item()
        if vl < best_vl:
            best_vl, best_st, wait = vl, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        _, _, pa_te, pb_te = model(xa_te, xb_te)

    pred_a, pred_b = pa_te.cpu().numpy(), pb_te.cpu().numpy()
    var_y = np.var(y_te); y_bar = np.mean(y_te)
    mse_a = mean_squared_error(y_te, pred_a)
    mse_b = mean_squared_error(y_te, pred_b)
    i_a = pointwise_info(y_te, pred_a, var_y, y_bar, mse_a)
    i_b = pointwise_info(y_te, pred_b, var_y, y_bar, mse_b)
    both_pos = (i_a > 0) & (i_b > 0)
    r_source = np.where(both_pos, np.minimum(i_a, i_b), 0.0)
    return float(np.mean(r_source))


# ============================================================
# 4. CASCADE
# ============================================================

def run_cascade(splits, modalities, device="cpu"):
    names = list(modalities.keys())
    base = "Demographics"
    rest = [m for m in names if m != base]

    y_tr, y_va, y_te = splits["train"]["y"], splits["val"]["y"], splits["test"]["y"]
    var_y = float(np.var(y_te))
    h_y = H_Y(var_y)

    print(f"\n  H(Y) = {h_y:.4f} nats,  Var(Y) = {var_y:.2f}")

    # -- Step 0 --
    print(f"\n{'=' * 60}\nStep 0: {base}\n{'=' * 60}")
    pred_base = fit_and_predict(
        splits["train"]["X"][base], y_tr,
        splits["val"]["X"][base], y_va,
        splits["test"]["X"][base],
    )
    r2_base = max(0.0, r2_score(y_te, pred_base))
    I_base = MI_from_r2(r2_base)
    ve_base = pid_nats_to_r2(0.0, r2_base, 0.0, r2_base)
    print(f"  R² = {r2_base:.4f},  I = {I_base:.4f} nats")

    results = [{
        "step": 0, "added": base, "accumulated": [base],
        "R2_joint": r2_base, "I_joint": I_base,
        "R_nats": 0, "U_old_nats": I_base, "U_new_nats": 0, "S_nats": 0,
        "R_source_nats": 0, "R_mech_nats": 0,
        "r2_R": 0, "r2_U_old": ve_base["r2_U_old"],
        "r2_U_new": 0, "r2_S": 0,
        "r2_R_source": 0, "r2_R_mech": 0,
        "r2_unexplained": ve_base["r2_unexplained"],
    }]

    accumulated = [base]

    for step in range(len(rest)):
        print(f"\n{'=' * 60}\nStep {step + 1}: Greedy selection\n{'=' * 60}")
        candidates = [m for m in rest if m not in accumulated]
        if not candidates: break

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
            print(f"  {cand:<22s}: R² = {r2:.4f}  (Δ = {delta:+.4f})")
            if r2 > best_r2:
                best_r2, best_mod = r2, cand

        print(f"\n  --> Adding: {best_mod}")
        accumulated.append(best_mod)

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

        pid = compute_pid(y_te, pred_old, pred_new, pred_joint)

        R_source_nats = train_gk(
            concat(splits["train"]["X"], accumulated[:-1]),
            splits["train"]["X"][best_mod], y_tr,
            concat(splits["val"]["X"], accumulated[:-1]),
            splits["val"]["X"][best_mod], y_va,
            concat(splits["test"]["X"], accumulated[:-1]),
            splits["test"]["X"][best_mod], y_te,
            align_w=10.0, device=device,
        )

        R_total_nats = max(pid["R"], R_source_nats)
        R_total_nats = min(R_total_nats, pid["I_old"], pid["I_new"])
        R_source_nats = min(R_source_nats, R_total_nats)

        U_old_nats = pid["I_old"] - R_total_nats
        U_new_nats = pid["I_new"] - R_total_nats
        S_nats     = pid["I_joint"] - pid["I_old"] - pid["I_new"] + R_total_nats

        if S_nats < 0:
            R_total_nats  -= S_nats
            R_source_nats -= S_nats
            R_source_nats  = min(R_source_nats, R_total_nats)
            U_old_nats = pid["I_old"] - R_total_nats
            U_new_nats = pid["I_new"] - R_total_nats
            S_nats = 0.0

        R_mech_nats = max(0.0, R_total_nats - R_source_nats)
        I_joint, R2_joint = pid["I_joint"], pid["R2_joint"]

        ve = pid_nats_to_r2(R_total_nats, pid["R2_old"], pid["R2_new"], R2_joint, R_source_nats)

        print(f"\n  R²_old   = {pid['R2_old']:.4f}  ({', '.join(accumulated[:-1])})")
        print(f"  R²_new   = {pid['R2_new']:.4f}  ({best_mod})")
        print(f"  R²_joint = {R2_joint:.4f}")
        print(f"\n  PID in nats:")
        print(f"    R       = {R_total_nats:.4f}  [src: {R_source_nats:.4f}, mech: {R_mech_nats:.4f}]")
        print(f"    U_old   = {U_old_nats:.4f}")
        print(f"    U_new   = {U_new_nats:.4f}")
        print(f"    S       = {S_nats:.4f}")
        print(f"\n  PID as fraction of Var(Y):")
        print(f"    R       = {ve['r2_R']:.4f}  [src: {ve['r2_R_source']:.4f}, mech: {ve['r2_R_mech']:.4f}]")
        print(f"    U_old   = {ve['r2_U_old']:.4f}")
        print(f"    U_new   = {ve['r2_U_new']:.4f}")
        print(f"    S       = {ve['r2_S']:.4f}")
        print(f"    Unexp.  = {ve['r2_unexplained']:.4f}")

        results.append({
            "step": step + 1, "added": best_mod,
            "accumulated": list(accumulated),
            "R2_joint": R2_joint, "I_joint": I_joint,
            "R_nats": R_total_nats, "U_old_nats": U_old_nats,
            "U_new_nats": U_new_nats, "S_nats": S_nats,
            "R_source_nats": R_source_nats, "R_mech_nats": R_mech_nats,
            "r2_R": ve["r2_R"], "r2_U_old": ve["r2_U_old"],
            "r2_U_new": ve["r2_U_new"], "r2_S": ve["r2_S"],
            "r2_R_source": ve["r2_R_source"], "r2_R_mech": ve["r2_R_mech"],
            "r2_unexplained": ve["r2_unexplained"],
        })

    return results


# ============================================================
# 5. VISUALIZATION
# ============================================================

def plot_cascade(results, save_path="cascading_pid_slope.pdf"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    SHORT = {
        "Demographics": "Demog",
        "Cognitive_Baseline": "CogBL",
        "CSF_Amyloid": "Aβ",
        "CSF_Tau": "Tau",
        "Volumetric_MRI": "MRI",
        "FDG_PET": "FDG",
    }
    def sn(mod): return SHORT.get(mod, mod)
    def sl(mods): return "+".join(sn(m) for m in mods)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    if n == 1: axes = [axes]

    C = {
        "R_source": "#E8593C", "R_mech": "#F2A623",
        "U_old": "#3B8BD4", "U_new": "#5DCAA5",
        "S": "#7F77DD", "unexplained": "#D3D1C7",
    }

    for i, (ax, r) in enumerate(zip(axes, results)):
        added = sn(r["added"])
        old_mods = sl(r["accumulated"][:-1]) if i > 0 else ""

        if i == 0:
            vals = [r["r2_U_old"], r["r2_unexplained"]]
            labs = [f'U({added})\n{r["r2_U_old"]:.1%}',
                    f'Unexplained\n{r["r2_unexplained"]:.1%}']
            cols = [C["U_old"], C["unexplained"]]
        else:
            vals = [r["r2_R_source"], r["r2_R_mech"],
                    r["r2_U_old"], r["r2_U_new"],
                    r["r2_S"], r["r2_unexplained"]]
            labs = [f'R_src({added},\n{old_mods})\n{r["r2_R_source"]:.1%}',
                    f'R_mech({added},\n{old_mods})\n{r["r2_R_mech"]:.1%}',
                    f'U({old_mods})\n{r["r2_U_old"]:.1%}',
                    f'U({added})\n{r["r2_U_new"]:.1%}',
                    f'S({added},\n{old_mods})\n{r["r2_S"]:.1%}',
                    f'Unexplained\n{r["r2_unexplained"]:.1%}']
            cols = [C["R_source"], C["R_mech"], C["U_old"],
                    C["U_new"], C["S"], C["unexplained"]]

        keep = [(v, l, c) for v, l, c in zip(vals, labs, cols) if v > 0.005]
        if keep: vals, labs, cols = zip(*keep)

        ax.pie(vals, labels=labs, colors=cols, startangle=90,
               textprops={"fontsize": 6},
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        title = f"Step {i}: +{added}" if i > 0 else f"Step 0: {added}"
        ax.set_title(f"{title}\nR²={r['R2_joint']:.3f}", fontsize=9, fontweight="bold")

    patches = [
        mpatches.Patch(color=C["R_source"], label="Redundancy (source)"),
        mpatches.Patch(color=C["R_mech"], label="Redundancy (mechanistic)"),
        mpatches.Patch(color=C["U_old"], label="Unique (accumulated)"),
        mpatches.Patch(color=C["U_new"], label="Unique (new modality)"),
        mpatches.Patch(color=C["S"], label="Synergy"),
        mpatches.Patch(color=C["unexplained"], label="Unexplained"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cascading PID — Predicting ADAS13 Slope (pts/yr) from Baseline",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close()


def print_summary(results):
    print(f"\n{'=' * 120}")
    print("CASCADING PID SUMMARY — Target: ADAS13 slope (pts/yr)")
    print(f"{'=' * 120}")
    print(
        f"{'Step':<5} {'Added':<22} {'R²':>6} "
        f"{'R':>7} {'R_src':>7} {'R_mch':>7} "
        f"{'U_old':>7} {'U_new':>7} {'S':>7} {'Unexp':>7}  "
        f"{'(nats: R':>8} {'U_o':>5} {'U_n':>5} {'S)':>5}"
    )
    print("-" * 120)
    for r in results:
        print(
            f"{r['step']:<5} {r['added']:<22} "
            f"{r['R2_joint']:>6.3f} "
            f"{r['r2_R']:>7.4f} {r.get('r2_R_source',0):>7.4f} {r.get('r2_R_mech',0):>7.4f} "
            f"{r['r2_U_old']:>7.4f} {r['r2_U_new']:>7.4f} {r['r2_S']:>7.4f} "
            f"{r['r2_unexplained']:>7.4f}  "
            f"{r['R_nats']:>8.4f} {r['U_old_nats']:>5.3f} {r['U_new_nats']:>5.3f} {r['S_nats']:>5.3f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tadpole-path", type=str, required=True)
    parser.add_argument("--adnimerge-path", type=str, required=True)
    parser.add_argument("--min-months", type=int, default=24,
                        help="Minimum follow-up in months for slope computation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str,
                        default="./cascading_pid_slope_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data and computing ADAS13 slopes...")
    df = load_data_slope(
        args.tadpole_path, args.adnimerge_path,
        min_months=args.min_months,
    )

    modalities = define_modalities()
    for name, feats in modalities.items():
        print(f"  {name}: {feats}")

    splits = prepare_data(df, modalities)
    results = run_cascade(splits, modalities, device=args.device)
    print_summary(results)

    results_ser = []
    for r in results:
        r_copy = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                r_copy[k] = float(v)
            elif isinstance(v, np.ndarray):
                r_copy[k] = v.tolist()
            else:
                r_copy[k] = v
        results_ser.append(r_copy)

    with open(os.path.join(args.output_dir, "cascade_slope_results.json"), "w") as f:
        json.dump(results_ser, f, indent=2)

    plot_cascade(results, save_path=os.path.join(args.output_dir, "cascading_pid_slope.pdf"))
    print("Done!")