# train_iforest.py — IsolationForest + RobustScaler (UNSW-NB15) + MLflow
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# MLflow helpers
from utils.mlflow_utils import setup_mlflow, start_run, log_params, log_metrics, log_artifacts
setup_mlflow(experiment_name="IDS_Training")  # même expérience que XGB et AE

# --- Chemins (adapte si besoin) ---
TRAIN_CSV = r"C:\Users\salsa\Desktop\Dataset\UNSW_NB15_training-set (1).csv"
TEST_CSV  = r"C:\Users\salsa\Desktop\Dataset\UNSW_NB15_testing-set (1).csv"

OUT_PATH     = r"C:\Users\salsa\Desktop\PFE\IDS\IDS\ids_iforest.joblib"
SUMMARY_PATH = r"C:\Users\salsa\Desktop\PFE\IDS\IDS\iforest_train_summary.txt"

# --- Features pour IForest (doivent exister en prod) ---
NUM_COLS_IF = ["dpkts", "sttl", "smean"]  # = stables côté app.py

print("Chargement UNSW...")
df_tr = pd.read_csv(TRAIN_CSV)
df_te = pd.read_csv(TEST_CSV)
df = pd.concat([df_tr, df_te], ignore_index=True)
df.columns = [c.lower().strip() for c in df.columns]

# --- Label binaire: normal = 0
label_col = "label" if "label" in df.columns else ("attack_cat" if "attack_cat" in df.columns else None)
if label_col is None:
    raise ValueError("Colonne de label introuvable (label / attack_cat).")

if label_col == "attack_cat":
    df[label_col] = np.where(df[label_col].astype(str).str.lower().eq("normal"), 0, 1)
else:
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

# --- Colonnes nécessaires pour construire smean
for c in ["sbytes", "spkts", "dpkts", "sttl"]:
    if c not in df.columns:
        df[c] = 0

df["smean"] = (pd.to_numeric(df.get("sbytes", 0), errors="coerce").fillna(0.0) /
               (pd.to_numeric(df.get("spkts", 0), errors="coerce").fillna(0.0) + 1e-6))

# --- Jeu "normal" pour apprendre le profil
norm = df[df[label_col] == 0].copy()
Xn = norm[NUM_COLS_IF].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# --- Pipeline: RobustScaler + IsolationForest
pipe = Pipeline([
    ("rs", RobustScaler()),
    ("if", IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=0.02,    # ≈2% anormaux côté train
        random_state=42,
        n_jobs=-1,
        verbose=0
    ))
])

print("Fit IsolationForest...")
pipe.fit(Xn)

# --- Score d'anomalie (plus grand => plus anormal)
scores = -pipe.named_steps["if"].score_samples(pipe.named_steps["rs"].transform(Xn))
# seuil tau ≈ 98e percentile => ~2% les plus anormaux (sur trafic normal)
tau = float(np.percentile(scores, 98.0))

# Sauvegarde artefact modèle
joblib.dump({
    "iforest": pipe,
    "tau": tau,
    "num_cols_if": NUM_COLS_IF
}, OUT_PATH)

print(f"✅ IForest sauvegardé -> {OUT_PATH}")
print(f"Seuil tau ≈ {tau:.6f} (≈2% anormaux sur trafic normal)")

# Résumé texte
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write(
        "=== IsolationForest IDS Training Summary ===\n"
        f"Date: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Train CSV: {TRAIN_CSV}\nTest  CSV: {TEST_CSV}\n\n"
        f"Features: {NUM_COLS_IF}\n"
        f"n_total: {len(df)} | n_normals_train: {len(Xn)} | n_attacks_total: {int((df[label_col]==1).sum())}\n\n"
        f"IForest: n_estimators=300, contamination=0.02, random_state=42\n"
        f"tau (98th pct on normal) = {tau:.6f}\n"
        f"train_scores_mean = {float(np.mean(scores)):.6f}\ntrain_scores_p95 = {float(np.percentile(scores,95)):.6f}\n"
    )

# --- Logging MLflow
with start_run(run_name="iforest_train"):
    log_params({
        "model": "IsolationForest",
        "dataset": "UNSW_NB15",
        "train_csv": TRAIN_CSV,
        "test_csv": TEST_CSV,
        "features": str(NUM_COLS_IF),
        "n_total": len(df),
        "n_normals_train": len(Xn),
        "n_attacks_total": int((df[label_col]==1).sum()),
        "n_estimators": 300,
        "contamination": 0.02,
        "random_state": 42
    })
    log_metrics({
        "tau_iforest": float(tau),
        "train_scores_mean": float(np.mean(scores)),
        "train_scores_p95": float(np.percentile(scores,95))
    })
    log_artifacts([OUT_PATH, SUMMARY_PATH], artifact_path="training")

print("✅ Entraînement IsolationForest terminé + log MLflow (IDS_Training).")
