import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── 1. Load & filter TADPOLE_D1_D2 ──────────────────────────────────────────
df_tadpole = pd.read_csv("/Users/robinlouiset/Documents/tadpole_challenge/TADPOLE_D1_D2.csv", low_memory=False)
df_tadpole = df_tadpole[df_tadpole["VISCODE"] == "bl"]
df_tadpole = df_tadpole[df_tadpole["FLDSTRENG"].notna()]
df_tadpole = df_tadpole[df_tadpole["FDG"].notna()]

# ── 2. Load & filter ADNIMERGE ───────────────────────────────────────────────
df_adni = pd.read_csv("/Users/robinlouiset/Documents/tadpole_challenge/ADNIMERGE.csv", low_memory=False)
df_adni = df_adni[df_adni["VISCODE"] == "bl"]
df_adni = df_adni[df_adni["ABETA_bl"].notna() & df_adni["TAU_bl"].notna() & df_adni["PTAU_bl"].notna()]

# ── 3. Merge on common subject identifier ────────────────────────────────────
df = pd.merge(df_tadpole, df_adni, on=["RID", "VISCODE"], how="inner", suffixes=("", "_adni"))

# ── 4. Keep AD, CN, and LMCI (relabeled as MCI) ─────────────────────────────
df = df[df["DX_bl"].isin(["AD", "CN", "LMCI"])]
df["DX_bl"] = df["DX_bl"].replace({"LMCI": "MCI"})
valid_labels = ["AD", "CN", "MCI"]

# ── 5. Define features & target ──────────────────────────────────────────────
volumes_to_normalize = ["Ventricles", "Hippocampus", "Entorhinal", "Fusiform", "MidTemp"]
clinical_features = ["ABETA_bl", "TAU_bl", "PTAU_bl", "AGE", "PTEDUCAT", "PTGENDER"]

target = "DX_bl"

# ── 6. Prepare X and y ──────────────────────────────────────────────────────
df["PTGENDER"] = df["PTGENDER"].map({"Male": 1, "Female": 0})

for col in ["ABETA_bl", "TAU_bl", "PTAU_bl", "ICV", "FDG"] + volumes_to_normalize:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Normalize volumes by ICV
for col in volumes_to_normalize:
    df[col + "_norm"] = df[col] / df["ICV"]

volumetric_features = [col + "_norm" for col in volumes_to_normalize]
all_features = clinical_features + volumetric_features + ["FDG"]

print(f"Total features: {len(all_features)}")
print(f"  {len(clinical_features)} clinical")
print(f"  {len(volumetric_features)} ICV-normalized volumetric")
print(f"  1 global FDG")

df_model = df[all_features + [target]].dropna()
print(f"\nSamples after dropping NaNs: {len(df_model)}")

# Print class distribution
print("\n── Class distribution ──")
class_counts = df_model[target].value_counts()
for label, count in class_counts.items():
    print(f"  {label}: {count} ({100 * count / len(df_model):.1f}%)")

# Encode labels
le = LabelEncoder()
le.fit(valid_labels)

X = df_model[all_features].values
y = le.transform(df_model[target].values)

print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 7. XGBoost classification with stratified 10-fold CV ─────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_child_weight=3,
    objective="multi:softprob",
    num_class=len(valid_labels),
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)

acc_scores, f1_scores = [], []
all_y_true, all_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    acc_scores.append(acc)
    f1_scores.append(f1)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Fold {fold:2d}: Accuracy={acc:.3f}  Weighted-F1={f1:.3f}")

print("\n── Cross-validation summary ──")
print(f"Accuracy    : {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
print(f"Weighted-F1 : {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

print("\n── Aggregated classification report (all folds) ──")
print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))

print("── Confusion matrix (rows=true, cols=pred) ──")
print(f"Classes: {list(le.classes_)}")
print(confusion_matrix(all_y_true, all_y_pred))

# ── 8. Feature importance ───────────────────────────────────────────────────
model.fit(X, y)
importances = model.feature_importances_
feat_imp = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
print("\n── Feature importance ──")
for name, imp in feat_imp:
    print(f"  {name:25s} {imp:.4f}")