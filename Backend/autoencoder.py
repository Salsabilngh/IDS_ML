# train_ae_mlp.py — Autoencoder (MLPRegressor) + RobustScaler (UNSW-NB15) + MLflow
import numpy as np, pandas as pd, joblib
from datetime import datetime

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# MLflow helpers (expérience partagée avec les autres modèles)
from utils.mlflow_utils import setup_mlflow, start_run, log_params, log_metrics, log_artifacts
setup_mlflow(experiment_name="IDS_Training")

# === Chemins (adapte si besoin) ===
TRAIN_CSV = r"C:\Users\salsa\Desktop\Dataset\UNSW_NB15_training-set (1).csv"
TEST_CSV  = r"C:\Users\salsa\Desktop\Dataset\UNSW_NB15_testing-set (1).csv"
OUT_PATH  = r"C:\Users\salsa\Desktop\PFE\IDS\IDS\ids_ae_mlp.joblib"
SUMMARY_PATH = r"C:\Users\salsa\Desktop\PFE\IDS\IDS\ae_mlp_train_summary.txt"

NUM_COLS_AE = ["dpkts","sttl","smean"]   # mêmes colonnes dispo côté app.py

print("Chargement UNSW...")
df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
df.columns = [c.lower().strip() for c in df.columns]

# Label binaire (normal=0, attaque=1)
label_col = "label" if "label" in df.columns else ("attack_cat" if "attack_cat" in df.columns else None)
if label_col is None:
    raise ValueError("Colonne de label introuvable (label / attack_cat).")
if label_col == "attack_cat":
    df[label_col] = np.where(df[label_col].astype(str).str.lower().eq("normal"), 0, 1)
else:
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

# smean = sbytes/spkts + garde-fous
for c in ["sbytes","spkts","dpkts","sttl"]:
    if c not in df.columns:
        df[c] = 0.0
df["smean"] = (pd.to_numeric(df["sbytes"], errors="coerce").fillna(0.0) /
               (pd.to_numeric(df["spkts"], errors="coerce").fillna(0.0) + 1e-6))

# Jeu normal pour l'apprentissage (reconstruction X->X)
norm = df[df[label_col]==0].copy()
Xn = norm[NUM_COLS_AE].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

# Scaler + "AE" MLP (reconstruction X->X)
scaler = RobustScaler()
Xn_s   = scaler.fit_transform(Xn)

# Architecture compacte (tu peux ajuster)
hidden = (8, 3, 8)  # goulot d’étranglement = 3
mlp = MLPRegressor(
    hidden_layer_sizes=hidden,
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=42,
    verbose=False
)

print("Fit MLPRegressor (autoencoder-like)...")
mlp.fit(Xn_s, Xn_s)   # target = Xn_s (reconstruction)

# Reconstruction sur train-normal & seuil tau (≈2% plus grosses erreurs)
Xhat_n = mlp.predict(Xn_s)
mse_n  = np.mean((Xhat_n - Xn_s)**2, axis=1)
tau    = float(np.percentile(mse_n, 98.0))

# Évaluation sur TOUT le dataset (utile pr AUC/PR-AUC, alert_rate@tau)
X_all = df[NUM_COLS_AE].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
Xs_all = scaler.transform(X_all)
Xhat_all = mlp.predict(Xs_all)
mse_all  = np.mean((Xhat_all - Xs_all)**2, axis=1)
y_all = df[label_col].values.astype(int)

# AUC: plus l'erreur est grande plus c'est "attaque"
try:
    auc = roc_auc_score(y_all, mse_all)
except Exception:
    auc = float("nan")
try:
    pr_auc = average_precision_score(y_all, mse_all)
except Exception:
    pr_auc = float("nan")

alert_rate_all = float(np.mean(mse_all >= tau))

# Sauvegarde artefact modèle
joblib.dump({
    "scaler": scaler,
    "mlp": mlp,
    "tau_mse": tau,
    "num_cols_ae": NUM_COLS_AE
}, OUT_PATH)

print(f"✅ AE-MLP sauvegardé -> {OUT_PATH}")
print(f"Seuil tau_mse ≈ {tau:.6f} (≈2% anormaux sur trafic normal)")
print(f"ROC-AUC(all)={auc:.4f} | PR-AUC(all)={pr_auc:.4f} | alert_rate@tau={alert_rate_all:.3f}")

# Résumé texte
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write(
        "=== AE-MLP IDS Training Summary ===\n"
        f"Date: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Train CSV: {TRAIN_CSV}\nTest  CSV: {TEST_CSV}\n\n"
        f"Features: {NUM_COLS_AE}\n"
        f"n_total: {len(df)} | n_normals_train: {len(Xn)} | n_attacks_total: {int((df[label_col]==1).sum())}\n\n"
        f"MLP hidden={hidden}, max_iter=200, activation=relu, solver=adam\n"
        f"tau_mse (98th pct on normal) = {tau:.6f}\n"
        f"ROC-AUC(all) = {auc:.6f}\nPR-AUC(all) = {pr_auc:.6f}\n"
        f"alert_rate@tau(all) = {alert_rate_all:.6f}\n"
        f"train_mse_mean = {float(np.mean(mse_n)):.6f}\ntrain_mse_p95 = {float(np.percentile(mse_n,95)):.6f}\n"
    )

# Logging MLflow
with start_run(run_name="ae_mlp_train"):
    log_params({
        "model": "AE-MLP (reconstruction)",
        "dataset": "UNSW_NB15",
        "train_csv": TRAIN_CSV,
        "test_csv": TEST_CSV,
        "features": str(NUM_COLS_AE),
        "n_total": len(df),
        "n_normals_train": len(Xn),
        "n_attacks_total": int((df[label_col]==1).sum()),
        "hidden_layers": str(hidden),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 200,
        "random_state": 42
    })
    log_metrics({
        "tau_mse": float(tau),
        "roc_auc_all": float(auc) if not np.isnan(auc) else 0.0,
        "pr_auc_all": float(pr_auc) if not np.isnan(pr_auc) else 0.0,
        "alert_rate_at_tau_all": alert_rate_all,
        "train_mse_mean": float(np.mean(mse_n)),
        "train_mse_p95": float(np.percentile(mse_n,95))
    })
    log_artifacts([OUT_PATH, SUMMARY_PATH], artifact_path="training")

print("✅ Entraînement AE-MLP terminé + log MLflow (IDS_Training).")
