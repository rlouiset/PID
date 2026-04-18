import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. Load & filter TADPOLE_D1_D2 ──────────────────────────────────────────
df_tadpole = pd.read_csv("/Users/robinlouiset/Documents/tadpole_challenge/TADPOLE_D1_D2.csv", low_memory=False)
df_tadpole = df_tadpole[df_tadpole["VISCODE"] == "bl"]
df_tadpole = df_tadpole[df_tadpole["FLDSTRENG"].notna()]
df_tadpole = df_tadpole[df_tadpole["FDG"].notna()]
df_tadpole = df_tadpole[df_tadpole["AV45"].notna()]

# ── 2. Load & filter ADNIMERGE ───────────────────────────────────────────────
df_adni = pd.read_csv("/Users/robinlouiset/Documents/tadpole_challenge/ADNIMERGE.csv", low_memory=False)
df_adni = df_adni[df_adni["VISCODE"] == "bl"]
df_adni = df_adni[df_adni["ABETA_bl"].notna() & df_adni["TAU_bl"].notna() & df_adni["PTAU_bl"].notna()]
df_adni = df_adni[df_adni["FDG"].notna() & df_adni["AV45"].notna()]

# ── 3. Merge on common subject identifier ────────────────────────────────────
df = pd.merge(df_tadpole, df_adni, on=["RID", "VISCODE"], how="inner", suffixes=("", "_adni"))

# ── 4. Define features & target ──────────────────────────────────────────────
volumes_to_normalize = ["Ventricles_bl", "Hippocampus_bl", "Entorhinal_bl", "Fusiform_bl", "MidTemp_bl"]
clinical_features = ["AGE", "PTEDUCAT", "PTGENDER", "APOE4", "ABETA_bl", "TAU_bl", "PTAU_bl"]

# 10 AD-relevant regional FDG-PET features (bilateral key regions)
pet_features = [
    # Precuneus (bilateral)
    "PRECUNL01_BAIPETNMRC_09_12_16",
    "PRCUNSR01_BAIPETNMRC_09_12_16",
    # Fusiform (bilateral)
    "FUSFRML01_BAIPETNMRC_09_12_16",
    "FUSFRMR01_BAIPETNMRC_09_12_16",
    # Parahippocampal (bilateral)
    "PARAHIPL01_BAIPETNMRC_09_12_16",
    "PARAHIPR01_BAIPETNMRC_09_12_16",
    # Hippocampus (bilateral)
    "HIPPL01_BAIPETNMRC_09_12_16",
    "HIPPR01_BAIPETNMRC_09_12_16",
    # Posterior cingulate (bilateral)
    "CINGPSTL01_BAIPETNMRC_09_12_16",
    "CINGPSTR01_BAIPETNMRC_09_12_16",
]

target = "ADAS13"

# ── 5. Prepare X and y ──────────────────────────────────────────────────────
df["PTGENDER"] = df["PTGENDER"].map({"Male": 1, "Female": 0})

for col in ["ABETA_bl", "TAU_bl", "PTAU_bl", "ICV", target] + volumes_to_normalize + pet_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Normalize volumes by ICV
for col in volumes_to_normalize:
    df[col + "_norm"] = df[col] / df["ICV"]

volumetric_features = [col + "_norm" for col in volumes_to_normalize]
all_features = clinical_features + volumetric_features + ["FDG"] # + pet_features

print(f"Total features: {len(all_features)}")
print(f"  {len(clinical_features)} clinical")
print(f"  {len(volumetric_features)} ICV-normalized volumetric")
print(f"  {len(pet_features)} regional FDG-PET")

df_model = df[all_features + [target]].dropna()
print(f"\nSamples after dropping NaNs: {len(df_model)}")
print(f"\n── Target ({target}) summary ──")
print(f"  Mean:   {df_model[target].mean():.2f}")
print(f"  Std:    {df_model[target].std():.2f}")
print(f"  Min:    {df_model[target].min():.2f}")
print(f"  Max:    {df_model[target].max():.2f}")
print(f"  Median: {df_model[target].median():.2f}")

X = df_model[all_features].values
y = df_model[target].values

# ── 6. XGBoost regression with 10-fold CV (90/10 splits) ────────────────────
kf = KFold(n_splits=10, shuffle=True, random_state=42)

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_child_weight=3,
    random_state=42,
    verbosity=0,
)

rmse_scores, mae_scores, r2_scores = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    print(f"Fold {fold:2d}: RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")

print("\n── Cross-validation summary ──")
print(f"RMSE : {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
print(f"MAE  : {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
print(f"R²   : {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# ── 7. Feature importance ───────────────────────────────────────────────────
model.fit(X, y)
importances = model.feature_importances_
feat_imp = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
print("\n── Feature importance (top 15) ──")
for name, imp in feat_imp[:15]:
    print(f"  {name:45s} {imp:.4f}")