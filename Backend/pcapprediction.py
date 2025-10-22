import os
import pandas as pd
import numpy as np
import joblib
import subprocess
import json

# --- CONFIG ---
PCAP_PATH    = r"C:\Users\salsa\Desktop\PFE\uploads\capture.pcap"
OUTPUT_CSV   = r"C:\Users\salsa\Desktop\PFE\uploads\flows_with_preds.csv"
MODEL_PATH   = r"C:\Users\salsa\Desktop\PFE\IDS\IDS\ids_best_pipeline.joblib"
FEATURES_PATH= r"C:\Users\salsa\Desktop\PFE\IDS\IDS\feature_list.json"

TSHARK_EXE   = r"C:\Program Files\Wireshark\tshark.exe"  # chemin complet = évite les soucis de PATH

PROTO_MAP = {6: "tcp", 17: "udp", 1: "icmp"}  # default: 'other'

# --- 1. Vérifier pcap ---
if not os.path.exists(PCAP_PATH):
    raise FileNotFoundError(f"❌ Fichier pcap introuvable : {PCAP_PATH}")
print(f"✅ Fichier détecté : {PCAP_PATH}")

# --- 2. Extraire paquets avec tshark ---
print("⏳ Extraction des paquets avec tshark...")
tshark_fields = [
    "-T", "fields",
    "-e", "frame.time_epoch",
    "-e", "ip.src",
    "-e", "ip.dst",
    "-e", "ip.proto",
    "-e", "frame.len",
    "-e", "tcp.srcport",
    "-e", "tcp.dstport",
    "-e", "udp.srcport",
    "-e", "udp.dstport"
]
cmd = [TSHARK_EXE, "-r", PCAP_PATH] + tshark_fields
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("tshark a échoué")

lines = result.stdout.strip().split("\n")
rows = [l.split("\t") for l in lines if l.strip() != ""]
df = pd.DataFrame(rows, columns=[
    "time", "src", "dst", "proto", "length",
    "tcp_sport", "tcp_dport", "udp_sport", "udp_dport"
])
print(f"✅ {len(df)} paquets extraits")

# --- 3. Nettoyage et types ---
# garder seulement lignes avec proto numérique
df = df[df["proto"].apply(lambda x: str(x).isdigit())].copy()
df["time"]   = pd.to_numeric(df["time"], errors="coerce").fillna(0.0)
df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(0.0)
df["proto"]  = pd.to_numeric(df["proto"], errors="coerce").fillna(-1).astype(int)

# choisir ports TCP sinon UDP, sinon 0
df["sport"] = np.where(df["tcp_sport"] != "", df["tcp_sport"], df["udp_sport"])
df["dport"] = np.where(df["tcp_dport"] != "", df["tcp_dport"], df["udp_dport"])
df.drop(columns=["tcp_sport","tcp_dport","udp_sport","udp_dport"], inplace=True)
df["sport"] = pd.to_numeric(df["sport"], errors="coerce").fillna(0).astype(int)
df["dport"] = pd.to_numeric(df["dport"], errors="coerce").fillna(0).astype(int)

# --- 4. Flows (agrégation 5-tuple) ---
flows = df.groupby(["src","dst","proto","sport","dport"]).agg(
    dur   = ("time", lambda x: x.max() - x.min()),
    spkts = ("time", "count"),
    sbytes= ("length", "sum"),
).reset_index()

# approximations simples
flows["dpkts"] = flows["spkts"]
flows["dbytes"] = flows["sbytes"]
flows["rate"]   = flows["sbytes"] / (flows["dur"] + 1e-6)

# --- 5. Colonnes catégorielles attendues par le modèle ---
# proto doit être string (tcp/udp/icmp/other)
flows["proto"] = flows["proto"].map(PROTO_MAP).fillna("other").astype(str)
# 'service' et 'state' n'existent pas ici -> utiliser placeholders string (pas numériques)
flows["service"] = "unknown"
flows["state"]   = "CON"  # état générique (placeholder)

print(f"✅ {len(flows)} flows agrégés")
print("DEBUG uniques proto:", flows["proto"].unique()[:10])
print("DEBUG dtypes avant mapping features:\n", flows.dtypes.head(20))

# --- 6. Charger modèle + feature_list ---
print("📦 Chargement du modèle IA...")
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    features = json.load(f)

# --- 7. Construire X avec les features attendues ---
X = pd.DataFrame(index=flows.index)
for feat in features:
    if feat in flows.columns:
        X[feat] = flows[feat]
    else:
        # valeurs par défaut
        if feat in ("proto","service","state"):
            X[feat] = "unknown"  # IMPORTANT: string, pas 0
        else:
            X[feat] = 0.0

# Forcer types: catégorielles en str, le reste en float
categorical_cols = ["proto","service","state"]
for c in categorical_cols:
    if c in X.columns:
        X[c] = X[c].astype(str)

for col in X.columns:
    if col not in categorical_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)

print("DEBUG dtypes après mapping features:\n", X.dtypes.head(20))

# --- 8. Prédiction ---
print("🤖 Prédiction en cours...")
try:
    probs = model.predict_proba(X)[:, 1]
except Exception as e:
    print("predict_proba a échoué, essai avec predict(). Raison:", repr(e))
    preds = model.predict(X)
    # si on tombe là, on crée une proba factice
    probs = np.where(preds==1, 0.75, 0.25)

flows["prob_attack"] = probs
flows["pred"] = (flows["prob_attack"] > 0.5).astype(int)
flows["status"] = np.where(flows["pred"] == 1, "ATTACK", "NORMAL")

# --- 9. Sauvegarde ---
flows.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Résultats enregistrés dans : {OUTPUT_CSV}")
print(flows[["src","dst","proto","service","state","status","prob_attack"]].head())
