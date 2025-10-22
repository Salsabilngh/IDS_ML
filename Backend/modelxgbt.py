# modelxgbt.py
# Entraînement XGBoost pour IDS (UNSW-NB15) — SANS classe custom.
# Pipeline: OneHotEncoder('proto') + RobustScaler(numériques) + XGBClassifier
# Sauvegardes (compatibles app.py) :
#   C:\Users\salsa\Desktop\PFE\IDS\IDS\ids_best_pipeline.joblib
#   C:\Users\salsa\Desktop\PFE\IDS\IDS\feature_list.json

import os, json, numpy as np, pandas as pd, joblib
from utils.mlflow_utils import setup_mlflow, start_run, log_params, log_metrics, log_artifacts
setup_mlflow(experiment_name="IDS_Training")  # crée/choisit l'expérience d'entraînement

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from datetime import datetime

# -------------------- PATHS --------------------
TRAIN_CSV = r"C:\Users\salsa\Desktop\Dataset\UNSW_NB15_training-set (1).csv"
TEST_CSV  = r"C:\Users\salsa\Desktop\Dataset\UNSW_NB15_testing-set (1).csv"

OUT_DIR = r"C:\Users\salsa\Desktop\PFE\IDS\IDS"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH        = os.path.join(OUT_DIR, "ids_best_pipeline.joblib")
FEATURE_LIST_PATH = os.path.join(OUT_DIR, "feature_list.json")
RUN_SUMMARY_PATH  = os.path.join(OUT_DIR, "xgb_train_summary.txt")

print("Chargement des CSV UNSW-NB15...")
df_tr = pd.read_csv(TRAIN_CSV)
df_te = pd.read_csv(TEST_CSV)
df = pd.concat([df_tr, df_te], ignore_index=True)
df.columns = [c.lower().strip() for c in df.columns]

# -------------------- Label binaire --------------------
label_col = 'label' if 'label' in df.columns else ('attack_cat' if 'attack_cat' in df.columns else None)
if label_col is None:
    raise ValueError("Colonne de label introuvable (label / attack_cat).")

if label_col == 'attack_cat':
    df[label_col] = np.where(df[label_col].astype(str).str.lower().eq('normal'), 0, 1)
else:
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)

# -------------------- Features alignées app.py --------------------
# NOTE: 'ct_srv_src' n'est pas calculée par app.py actuellement.
#       Si absente au moment de l'inférence, app.py mettra 0.0 pour cette feature.
num_cols = ['dpkts','sttl','smean','ct_srv_src','total_syn_count','syn_count_dst','malformed_ratio']
cat_cols = ['proto']  # encodée via OneHotEncoder

# Créer les colonnes manquantes (0) puis smean si possible
for c in set(num_cols + cat_cols + ['sbytes','spkts']):
    if c not in df.columns:
        df[c] = 0

df['smean'] = (pd.to_numeric(df.get('sbytes',0), errors='coerce').fillna(0.0) /
               (pd.to_numeric(df.get('spkts',0), errors='coerce').fillna(0.0) + 1e-6))

# Types
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
df['proto'] = df['proto'].astype(str).fillna('unknown')

use_cols = num_cols + cat_cols + [label_col]
df = df[use_cols].copy()

X = df[num_cols + cat_cols]
y = df[label_col]

# -------------------- Split --------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- Pipeline --------------------
pre = ColumnTransformer(
    transformers=[
        ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num_rs", RobustScaler(), num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

base_clf = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
    random_state=42,
)

pipe = Pipeline([("pre", pre), ("clf", base_clf)])

# Déséquilibre classes
neg, pos = int((y_train==0).sum()), int((y_train==1).sum())
spw = max(1.0, neg / max(1, pos))
pipe.set_params(clf__scale_pos_weight=spw)
print(f"Scale pos weight = {spw:.2f}")

# -------------------- Tuning --------------------
param_dist = {
    "clf__n_estimators": randint(200, 800),
    "clf__max_depth": randint(3, 12),
    "clf__learning_rate": uniform(0.01, 0.2),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
}
randcv = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=50,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)
print("🔎 RandomizedSearchCV (this may take long)...")
randcv.fit(X_train, y_train)
best_pipe = randcv.best_estimator_
bp = randcv.best_params_
print("Random best params:", bp)

def clip01(x, lo=0.5, hi=1.0):
    return float(min(hi, max(lo, float(x))))

pg = {
    "clf__n_estimators": [int(max(100, bp["clf__n_estimators"] - 50)),
                          int(bp["clf__n_estimators"]),
                          int(bp["clf__n_estimators"] + 50)],
    "clf__max_depth": [int(max(3, bp["clf__max_depth"] - 2)),
                       int(bp["clf__max_depth"]),
                       int(bp["clf__max_depth"] + 2)],
    "clf__learning_rate": [max(0.005, float(bp["clf__learning_rate"]) * 0.8),
                           float(bp["clf__learning_rate"]),
                           min(0.4, float(bp["clf__learning_rate"]) * 1.2)],
    "clf__subsample": [clip01(float(bp["clf__subsample"]) - 0.1),
                       clip01(float(bp["clf__subsample"])),
                       clip01(float(bp["clf__subsample"]) + 0.1)],
    "clf__colsample_bytree": [clip01(float(bp["clf__colsample_bytree"]) - 0.1),
                              clip01(float(bp["clf__colsample_bytree"])),
                              clip01(float(bp["clf__colsample_bytree"]) + 0.1)],
}
gridcv = GridSearchCV(
    estimator=best_pipe,
    param_grid=pg,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=2
)
print("🔧 GridSearchCV (affinage)...")
gridcv.fit(X_train, y_train)
best_pipe = gridcv.best_estimator_
gp = gridcv.best_params_
print("Grid best params:", gp)

# -------------------- Évaluation + seuil optimal --------------------
y_pred  = best_pipe.predict(X_val)
proba   = best_pipe.predict_proba(X_val)[:, 1]
acc  = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
rec  = recall_score(y_val, y_pred, zero_division=0)
f1   = f1_score(y_val, y_pred, zero_division=0)
roc  = roc_auc_score(y_val, proba)
pra  = average_precision_score(y_val, proba)

print(f"VAL  ACC={acc:.3f} | P={prec:.3f} | R={rec:.3f} | F1={f1:.3f} | ROC-AUC={roc:.3f} | PR-AUC={pra:.3f}")

precs, recs, thrs = precision_recall_curve(y_val, proba)
f1s = (2*(precs*recs)/(precs+recs+1e-12))
best_idx = int(np.nanargmax(f1s))
best_thr = float(thrs[max(0, best_idx-1)]) if len(thrs) else 0.5
print(f"Seuil F1 optimal ≈ {best_thr:.3f}")

# -------------------- Sauvegardes locales --------------------
feature_list = num_cols + cat_cols  # ordre strict attendu par app.py
with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_list, f)

joblib.dump({"pipeline": best_pipe, "best_threshold": best_thr}, MODEL_PATH)
print("💾 Modèle sauvegardé ->", MODEL_PATH)
print("💾 Feature list     ->", FEATURE_LIST_PATH)
print("ℹ️ Features finales utilisées:", feature_list)

# Petit résumé textuel (utile comme artefact)
with open(RUN_SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write(
        "=== XGB IDS Training Summary ===\n"
        f"Date: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Train CSV: {TRAIN_CSV}\nTest  CSV: {TEST_CSV}\n"
        f"n_samples_total: {len(df)} | n_features: {len(feature_list)}\n"
        f"num_cols: {num_cols}\ncat_cols: {cat_cols}\n"
        f"scale_pos_weight: {spw:.4f}\n\n"
        f"RandomizedSearch best: {bp}\n"
        f"GridSearch best:       {gp}\n\n"
        f"VAL metrics:\n"
        f"  ACC={acc:.4f}\n  PREC={prec:.4f}\n  RECALL={rec:.4f}\n"
        f"  F1={f1:.4f}\n  ROC_AUC={roc:.4f}\n  PR_AUC={pra:.4f}\n"
        f"Best threshold (F1): {best_thr:.4f}\n"
    )

# -------------------- Logging MLflow --------------------
with start_run(run_name="xgb_train"):
    # 1) log hyperparams / infos
    log_params({
        "model": "XGBClassifier",
        "dataset": "UNSW_NB15",
        "train_csv": TRAIN_CSV,
        "test_csv": TEST_CSV,
        "n_samples_total": len(df),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_features": len(feature_list),
        "num_cols": str(num_cols),
        "cat_cols": str(cat_cols),
        "scale_pos_weight": spw,
        "random_best_params": str(bp),
        "grid_best_params": str(gp),
    })

    # 2) log metrics
    log_metrics({
        "val_acc": acc,
        "val_precision": float(prec),
        "val_recall": float(rec),
        "val_f1": float(f1),
        "val_roc_auc": float(roc),
        "val_pr_auc": float(pra),
        "best_threshold": float(best_thr),
    })

    # 3) log artefacts (modèle + features + résumé)
    log_artifacts([MODEL_PATH, FEATURE_LIST_PATH, RUN_SUMMARY_PATH], artifact_path="training")

print("✅ Entraînement terminé et loggé dans MLflow (expérience: IDS_Training).")
