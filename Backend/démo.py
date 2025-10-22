# demo.py
# Génère IDS/IDS/X_test_unscaled.npy compatible (9 colonnes) pour le streaming

import os
import numpy as np
import pandas as pd

FEATURES = ['dur','spkts','sbytes','dpkts','dbytes','rate','proto','service','state']
N = 1000  # nombre d’échantillons de démo

def make_demo_samples(features, n):
    # base num (n x nb_features) initialisée à zéro
    df = pd.DataFrame(np.zeros((n, len(features))), columns=features)

    # renseigner les colonnes catégorielles en chaînes
    if 'proto'   in df.columns: df['proto']   = np.random.choice(['tcp','udp','icmp'], size=n, p=[0.7,0.25,0.05])
    if 'service' in df.columns: df['service'] = np.random.choice(['http','https','dns','ssh','ftp','unknown'], size=n)
    if 'state'   in df.columns: df['state']   = np.random.choice(['CON','FIN','INT','RST'], size=n)

    # un peu de variété réaliste sur les numériques
    if 'dur'    in df.columns: df['dur']    = np.random.exponential(0.2, size=n)   # 0..quelques secondes
    if 'spkts'  in df.columns: df['spkts']  = np.random.randint(1, 40, size=n)
    if 'sbytes' in df.columns: df['sbytes'] = np.random.randint(60, 20000, size=n)
    if 'dpkts'  in df.columns: df['dpkts']  = np.random.randint(1, 40, size=n)
    if 'dbytes' in df.columns: df['dbytes'] = np.random.randint(60, 20000, size=n)
    if 'rate'   in df.columns:
        # bytes/sec approximatif (évite div by 0)
        df['rate'] = (df['sbytes'] / (df['dur'] + 1e-6)).clip(upper=1e6)

    # s’assure que types sont corrects
    for c in ['proto','service','state']:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df[features].to_numpy()

def main():
    os.makedirs("IDS/IDS", exist_ok=True)
    X = make_demo_samples(FEATURES, N)
    np.save("IDS/IDS/X_test_unscaled.npy", X)
    print(f"✅ Généré: IDS/IDS/X_test_unscaled.npy | shape={X.shape}")

if __name__ == "__main__":
    main()

