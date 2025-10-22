# build_index.py
import os, glob, joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

UPLOAD_DIR = r"C:\Users\salsa\Desktop\PFE\uploads"  # adapte si besoin
OUT_DIR = UPLOAD_DIR
FEATURES_NUMERIC = None  # si None on déduira automatiquement des CSVs

# 1) Lire tous les *_with_preds.csv de uploads
pattern = os.path.join(UPLOAD_DIR, "*_with_preds.csv")
files = glob.glob(pattern)
print("Found:", len(files), "files")

rows = []
for f in files:
    try:
        df = pd.read_csv(f)
        # normaliser noms de colonnes si nécessaire
        df['source_file'] = os.path.basename(f)
        rows.append(df)
    except Exception as e:
        print("skip", f, e)

if len(rows) == 0:
    raise SystemExit("No flows CSV found in uploads/*. Exporte d'abord des fichiers _with_preds.csv")

data = pd.concat(rows, ignore_index=True)
data.reset_index(inplace=True)
data.rename(columns={'index':'flow_index'}, inplace=True)

# 2) Choisir features numériques pour l'embedding
if FEATURES_NUMERIC is None:
    # heuristique : toutes colonnes numériques sauf prob_attack/pred maybe keep them
    numeric = data.select_dtypes(include=[np.number]).columns.tolist()
    # drop identifiers that are not features
    drop_cols = {'flow_index'}
    FEATURES_NUMERIC = [c for c in numeric if c not in drop_cols]
print("Numeric features used for embeddings:", FEATURES_NUMERIC)

# 3) Build matrix
Xnum = data[FEATURES_NUMERIC].fillna(0.0).astype(float).values

# 4) Scaler + PCA
scaler = StandardScaler()
Xs = scaler.fit_transform(Xnum)

# keep dims (min(50, n_features))
n_comp = min(50, Xs.shape[1])
pca = PCA(n_components=n_comp, random_state=42)
Xred = pca.fit_transform(Xs)

# 5) Index (NearestNeighbors)
nn = NearestNeighbors(n_neighbors=50, algorithm='auto', metric='euclidean', n_jobs=-1)
nn.fit(Xred)

# 6) Save objects and meta
joblib.dump(scaler, os.path.join(OUT_DIR, "rag_scaler.joblib"))
joblib.dump(pca, os.path.join(OUT_DIR, "rag_pca.joblib"))
joblib.dump(nn, os.path.join(OUT_DIR, "rag_nn_index.joblib"))

meta_cols = ['flow_index','source_file','src','dst','proto','sport','dport','status','prob_attack']
meta = data[[c for c in meta_cols if c in data.columns]].copy()
meta.to_csv(os.path.join(OUT_DIR, "rag_index_meta.csv"), index=False)

print("Index built. Saved: rag_scaler.joblib, rag_pca.joblib, rag_nn_index.joblib, rag_index_meta.csv")
print("Total indexed flows:", Xred.shape[0])
