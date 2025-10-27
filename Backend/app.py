# app.py — IDS Ensemble (XGBoost + IsolationForest + AE-MLP)
# Streaming réaliste + Upload PCAP + MLflow

import os, json, time, traceback, subprocess, shutil, random
import numpy as np
import pandas as pd
import joblib

# ============== Eventlet (optionnel) =================
USE_EVENTLET = False  # True -> eventlet pour un stream plus fluide
if USE_EVENTLET:
    import eventlet
    eventlet.monkey_patch()

# ============== MLflow ===============================
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_INFER_EXP = "IDS_Inference"
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
mlflow.set_experiment(MLFLOW_INFER_EXP)
_ml_client = MlflowClient()
_STREAM_RUN_ID = None

def _mlflow_start_stream_run(run_name: str, params: dict):
    """Ouvre un run MLflow pour le streaming."""
    global _STREAM_RUN_ID
    try:
        exp = mlflow.get_experiment_by_name(MLFLOW_INFER_EXP)
        exp_id = exp.experiment_id if exp else mlflow.create_experiment(MLFLOW_INFER_EXP)
        run = _ml_client.create_run(experiment_id=exp_id, tags={"mlflow.runName": run_name})
        _STREAM_RUN_ID = run.info.run_id
        for k, v in (params or {}).items():
            _ml_client.log_param(_STREAM_RUN_ID, k, str(v))
    except Exception as e:
        print("[MLflow] start stream run error:", e)
        _STREAM_RUN_ID = None

def _mlflow_log_metric(name: str, value: float, step: int):
    if _STREAM_RUN_ID is None:
        return
    try:
        _ml_client.log_metric(_STREAM_RUN_ID, name, float(value), step=step)
    except Exception as e:
        print("[MLflow] log metric error:", e)

def _mlflow_end_stream_run():
    global _STREAM_RUN_ID
    _STREAM_RUN_ID = None

def mlflow_log_inference(run_name: str, params: dict, metrics: dict, artifacts: list):
    """Petit helper pour logger l'upload PCAP (run ponctuel)."""
    try:
        mlflow.set_experiment(MLFLOW_INFER_EXP)
        with mlflow.start_run(run_name=run_name):
            for k, v in (params or {}).items():
                mlflow.log_param(k, v)
            for k, v in (metrics or {}).items():
                mlflow.log_metric(k, float(v))
            if artifacts:
                for path in artifacts:
                    try:
                        mlflow.log_artifact(path, artifact_path="inference")
                    except Exception:
                        pass
    except Exception as e:
        print("[MLflow] inference log error:", e)

# ============== Flask + Socket.IO ====================
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Event
from werkzeug.utils import secure_filename

from flask_socketio import SocketIO
if USE_EVENTLET:
    socketio = SocketIO(cors_allowed_origins="*", async_mode='eventlet')
else:
    socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}})
socketio.init_app(app)

# ============== CONFIG ===============================
MODEL_PATH        = r"/app/IDS/IDS/ids_best_pipeline.joblib"
FEATURE_LIST_PATH = r"/app/IDS/IDS/feature_list.json"
TEST_DATA_PATH    = r"/app/IDS/IDS/X_test_unscaled.npy"

IFOREST_PATH     = r"/app/IDS/IDS/ids_iforest.joblib"
AE_PATH          = r"/app/IDS/IDS/ids_ae_mlp.joblib"

SLEEP_TIME        = 2
MAX_SIM_SAMPLES   = 5000

TSHARK_EXE = (
    os.environ.get("TSHARK_EXE")
    or shutil.which("tshark")
    
)
UPLOAD_DIR  = r"/app/uploads"
ALLOWED_EXT = {"pcap", "pcapng"}
PROTO_MAP   = {6: "tcp", 17: "udp", 1: "icmp"}
TSHARK_TIMEOUT_SEC = 120

# Seuils signatures (simples)
THRESH_SYN_GLOBAL          = 200
THRESH_UDP_PER_DST         = 200
THRESH_ICMP_PER_DST        = 100
THRESH_PORTS_PER_SRC_SCAN  = 50
MALFORMED_RATIO_ALERT      = 0.05
SYN_PER_DST_THRESH_ENV     = int(os.environ.get("SYN_PER_DST_THRESH", "5"))

# ============== ETAT GLOBAL ==========================
thread = None
thread_stop_event = Event()
thread_stop_event.set()
current_index = 0

pipe = None
feature_list = None
best_threshold = 0.5
X_test_unscaled = None  # npy (si compatible)

# IForest / AE
iforest_pack = None; tau_iforest = None; num_cols_if = ["dpkts","sttl","smean"]
ae_pack = None;      tau_mse     = None; num_cols_ae = ["dpkts","sttl","smean"]

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ============== CHARGEMENT ARTEFACTS =================
def load_artifacts():
    global pipe, feature_list, best_threshold, X_test_unscaled
    try:
        if os.path.exists(MODEL_PATH):
            md = joblib.load(MODEL_PATH)
            if isinstance(md, dict) and "pipeline" in md:
                pipe = md["pipeline"]; best_threshold = float(md.get("best_threshold", 0.5))
            else:
                pipe = md; best_threshold = 0.5
            log(f"✅ Modèle chargé: {MODEL_PATH} (threshold={best_threshold:.3f})")
        else:
            log(f"🚨 MODEL_PATH introuvable: {MODEL_PATH}")

        if os.path.exists(FEATURE_LIST_PATH):
            with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
                _fl = json.load(f)
            # On impose l'ordre attendu train (num puis cat) — garde tel quel
            feature_list = _fl
            log(f"✅ Feature list chargée ({len(feature_list)} features).")
        else:
            log(f"🚨 FEATURE_LIST_PATH introuvable: {FEATURE_LIST_PATH}")

        if os.path.exists(TEST_DATA_PATH):
            X_full = np.load(TEST_DATA_PATH, allow_pickle=True)
            X_test_unscaled = X_full[:MAX_SIM_SAMPLES] if len(X_full) > MAX_SIM_SAMPLES else X_full
            log(f"✅ X_test_unscaled chargé ({len(X_test_unscaled)} samples).")

        # Warmup
        if pipe is not None and feature_list is not None:
            warm = {c: ("unknown" if c in ("proto","service","state") else 0.0) for c in feature_list}
            try:
                _ = pipe.predict(pd.DataFrame([warm]))[0]
                log("🧪 Warmup OK (predict).")
            except Exception as e:
                log(f"🧪 Warmup FAILED: {e}")
        else:
            log("⚠️ Artefacts incomplets (pipe/feature_list manquants).")
    except Exception as e:
        log(f"🚨 Erreur de chargement artefacts: {e}")
        traceback.print_exc()

def load_iforest():
    global iforest_pack, tau_iforest, num_cols_if
    try:
        if os.path.exists(IFOREST_PATH):
            pack = joblib.load(IFOREST_PATH)
            iforest_pack = pack.get("iforest"); tau_iforest = float(pack.get("tau", 0.0))
            num_cols_if = list(pack.get("num_cols_if", num_cols_if))
            if iforest_pack is not None:
                log(f"✅ IsolationForest chargé | tau≈{tau_iforest:.6f} | cols={num_cols_if}")
    except Exception as e:
        log(f"⚠️ IForest load error: {e}")

def load_ae():
    global ae_pack, tau_mse, num_cols_ae
    try:
        if os.path.exists(AE_PATH):
            pk = joblib.load(AE_PATH)
            ae_pack = pk; tau_mse = float(pk.get("tau_mse", 0.0))
            num_cols_ae = list(pk.get("num_cols_ae", num_cols_ae))
            log(f"✅ AE-MLP chargé | tau_mse≈{tau_mse:.6f} | cols={num_cols_ae}")
    except Exception as e:
        log(f"⚠️ AE load error: {e}")

load_artifacts(); load_iforest(); load_ae()

# ============== PRE-FLIGHT ===========================
def _preflight():
    if not os.path.exists(TSHARK_EXE):
        log(f"ℹ️ TShark non trouvé: {TSHARK_EXE} (OK si stream seulement)")
    else:
        try:
            out = subprocess.run([TSHARK_EXE, "-v"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                log("✅ TShark OK: " + (out.stdout or out.stderr).splitlines()[0])
        except Exception as e:
            log(f"🚨 Erreur TShark: {e}")
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(os.path.join(UPLOAD_DIR, ".write_test"), "w", encoding="utf-8") as fh: fh.write("ok")
        os.remove(os.path.join(UPLOAD_DIR, ".write_test"))
        log(f"✅ UPLOAD_DIR OK: {UPLOAD_DIR}")
    except Exception as e:
        log(f"🚨 UPLOAD_DIR non accessible: {e}")
_preflight()

# ============== STREAM SOURCES =======================
def _row_from_npy(i: int):
    """Construit une ligne depuis X_test_unscaled si shape == len(feature_list)."""
    try:
        if X_test_unscaled is None: return None
        row = X_test_unscaled[i]
        if isinstance(row, dict):
            d = {c: row.get(c, ("unknown" if c in ("proto","service","state") else 0.0)) for c in feature_list}
            return pd.DataFrame([d])
        if hasattr(row, "__len__") and len(row) == len(feature_list):
            d = {}
            for c, v in zip(feature_list, row):
                d[c] = str(v) if c in ("proto","service","state") else float(v)
            return pd.DataFrame([d])
    except Exception:
        pass
    return None

def _row_synthetic(i: int):
    """Génère une ligne synthétique cohérente; 1 attaque sur 5."""
    d = {c: 0.0 for c in feature_list}
    base = {
        "dpkts": random.randint(3, 30),
        "sttl": random.choice([48, 50, 52, 60, 64, 255]),
        "smean": random.uniform(20, 200),
        "ct_srv_src": random.randint(0, 3),
        "total_syn_count": random.randint(0, 2),
        "syn_count_dst": random.randint(0, 2),
        "malformed_ratio": random.uniform(0.0, 0.02),
        "proto": random.choice(["tcp", "udp", "icmp"]),
    }
    for k, v in base.items():
        if k in d: d[k] = v
    if "proto" in d: d["proto"] = base["proto"]

    is_attack = ((i + 1) % 5 == 0)
    if is_attack:
        d["dpkts"] = random.randint(120, 400)
        d["smean"] = random.uniform(500, 5000)
        d["total_syn_count"] = random.randint(400, 1000)
        d["syn_count_dst"] = max(SYN_PER_DST_THRESH_ENV + random.randint(3, 10), 10)
        d["malformed_ratio"] = random.uniform(0.06, 0.3)
        d["ct_srv_src"] = random.randint(8, 25)
        d["sttl"] = random.choice([44, 46, 48, 250])
        d["proto"] = random.choice(["tcp", "udp"])

    for c in feature_list:
        if c in ("service","state") and c not in d:
            d[c] = "unknown"
        if c == "proto" and c not in d:
            d[c] = "tcp"
    return pd.DataFrame([d]), is_attack

def _make_stream_row(i: int):
    r = _row_from_npy(i)
    if r is not None:
        return r, False, "real"
    r, atk = _row_synthetic(i)
    return r, atk, "synthetic"

# ============== HELPERS =============================
def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def _save_text(path: str, text: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: f.write(text)
    except Exception as e:
        log(f"DEBUG: cannot write {path}: {e}")

# ============== PREDICTION (1 ligne) =================
def _predict_one(row_df: pd.DataFrame):
    """Renvoie (pred, prob_ensemble, prob_xgb, anom_if, ae_mse)"""
    try:
        prob_xgb = float(pipe.predict_proba(row_df)[0, 1])
    except Exception:
        pred_tmp = int(pipe.predict(row_df)[0])
        prob_xgb = 0.85 if pred_tmp else 0.15

    # IForest
    if iforest_pack is not None:
        try:
            X_if = row_df[num_cols_if].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            anom_if = -iforest_pack.named_steps["if"].score_samples(
                iforest_pack.named_steps["rs"].transform(X_if)
            )[0]
        except Exception:
            anom_if = 0.0
    else:
        anom_if = 0.0

    # AE
    if ae_pack is not None:
        try:
            X_ae = row_df[num_cols_ae].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
            scaler = ae_pack["scaler"]; mlp = ae_pack["mlp"]
            Xs = scaler.transform(X_ae); Xhat = mlp.predict(Xs)
            ae_mse = float(np.mean((Xhat - Xs)**2))
        except Exception:
            ae_mse = 0.0
    else:
        ae_mse = 0.0

    def _norm(v, tau):
        if tau is None or tau <= 0: return 0.0
        return float(min(1.0, max(0.0, (v - tau)/(tau + 1e-6))))

    anom_if01 = _norm(anom_if, tau_iforest)
    ae01      = _norm(ae_mse,  tau_mse)

    alpha, beta, gamma = 0.7, 0.2, 0.1
    prob_ens = alpha*prob_xgb + beta*anom_if01 + gamma*ae01
    pred = 1 if prob_ens >= best_threshold else 0
    return pred, prob_ens, prob_xgb, anom_if, ae_mse

# ============== THREAD STREAM =======================
def background_simulation_thread():
    log("▶️  Simulation (STREAM)")
    global current_index, thread
    i = current_index
    seen = 0; attacks = 0
    mean_xgb = 0.0; mean_ens = 0.0
    _mlflow_start_stream_run(
        run_name="stream:online",
        params={"threshold": best_threshold, "sleep_s": SLEEP_TIME}
    )
    try:
        while not thread_stop_event.is_set():
            row_df, _, mode_str = _make_stream_row(i)
            # forcer colonnes & ordre
            for c in feature_list:
                if c not in row_df.columns:
                    row_df[c] = ("unknown" if c in ("proto","service","state") else 0.0)
            row_df = row_df[feature_list]

            pred, p_ens, p_xgb, a_if, a_mse = _predict_one(row_df)
            status = "ATTACK" if pred==1 else "NORMAL"
            seen += 1; attacks += int(pred==1)
            mean_xgb = ((mean_xgb*(seen-1)) + p_xgb)/seen
            mean_ens = ((mean_ens*(seen-1)) + p_ens)/seen

            socketio.emit('ids_update', {
                "id": i+1, "status": status, "timestamp": time.time(),
                "feature_count": len(feature_list or []),
                "prob_attack": float(p_ens),
                "prob_xgb": float(p_xgb),
                "anom_score": float(a_if),
                "ae_mse": float(a_mse),
                "src": "-", "dst": "-",
                "proto": str(row_df.at[0,"proto"]) if "proto" in row_df else "-",
                "sport": 0, "dport": 0
            }, namespace='/ids')

            # MLflow
            _mlflow_log_metric("stream_seen", seen, step=i+1)
            _mlflow_log_metric("stream_attacks", attacks, step=i+1)
            _mlflow_log_metric("stream_attack_rate", attacks/max(1, seen), step=i+1)
            _mlflow_log_metric("stream_mean_prob_xgb", mean_xgb, step=i+1)
            _mlflow_log_metric("stream_mean_prob_ensemble", mean_ens, step=i+1)

            current_index = i; i += 1
            if USE_EVENTLET:
                eventlet.sleep(SLEEP_TIME)
            else:
                time.sleep(SLEEP_TIME)
    except Exception as e:
        log(f"❌ Exception STREAM: {e}")
        traceback.print_exc()
    finally:
        socketio.emit('status_control', {'state': 'IDLE', 'message': 'Simulation stoppée.'}, namespace='/ids')
        _mlflow_end_stream_run()
        log("⏹️  Simulation arrêtée.")
        thread = None

# ============== SIGNATURES (robuste DF) =============
def detect_signatures_from_df(_df: pd.DataFrame):
    """
    Version robuste qui ne dépend pas de la présence obligatoire des colonnes.
    Utilise un signal très simple: compter lignes TCP visibles (approx SYN).
    """
    alerts = []
    tcp_sport = _df["tcp_sport"] if "tcp_sport" in _df.columns else pd.Series([], dtype=str)
    total_syn = int((tcp_sport.astype(str) != "").sum())

    if total_syn >= THRESH_SYN_GLOBAL:
        alerts.append(("SYN flood global", "multiple", 0))

    agg = {
        "total_syn": total_syn,
        "malformed_ratio": 0.0,
        "total_pkts": int(len(_df))
    }
    return alerts, agg

# ============== SOCKET.IO HANDLERS ==================
@socketio.on('start_simulation', namespace='/ids')
def start_simulation(_=None):
    global thread
    if thread is not None and hasattr(thread, "is_alive") and not thread.is_alive():
        thread = None
    if not thread_stop_event.is_set() and thread is not None:
        return
    thread_stop_event.clear()
    thread = socketio.start_background_task(background_simulation_thread)
    socketio.emit('status_control', {'state': 'RUNNING', 'message': 'Stream actif...'}, namespace='/ids')

@socketio.on('stop_simulation', namespace='/ids')
def stop_simulation(_=None):
    global thread
    thread_stop_event.set()
    socketio.emit('status_control', {'state': 'STOPPING', 'message': 'Arrêt en cours...'}, namespace='/ids')
    thread = None

# ============== ROUTES ==============================
@app.route('/')
def index():
    return 'CyberWatch AI backend OK.'

@socketio.on('connect', namespace='/ids')
def handle_connect(_=None):
    log('Client connecté /ids')
    state = 'IDLE' if thread_stop_event.is_set() else 'RUNNING'
    msg = 'Ready. Press Start.' if state == 'IDLE' else 'Stream actif...'
    socketio.emit('status_control', {'state': state, 'message': msg}, namespace='/ids')

@socketio.on('disconnect', namespace='/ids')
def handle_disconnect(_=None):
    log('Client déconnecté /ids')

@app.route('/model_info')
def model_info():
    try:
        info = {
            "ok": True,
            "model_path": MODEL_PATH,
            "feature_list_path": FEATURE_LIST_PATH,
            "best_threshold": float(best_threshold),
            "n_features": len(feature_list) if feature_list else 0,
            "feature_list": feature_list,
            "iforest_loaded": bool(iforest_pack is not None),
            "iforest_tau": float(tau_iforest) if tau_iforest is not None else None,
            "iforest_cols": num_cols_if,
            "ae_loaded": bool(ae_pack is not None),
            "ae_tau_mse": float(tau_mse) if tau_mse is not None else None,
            "ae_cols": num_cols_ae,
        }
        return jsonify(info)
    except Exception as e:
        return jsonify(ok=False, error=repr(e)), 500

# ============== UPLOAD PCAP =========================
@app.route('/upload_pcap', methods=['POST'])
def upload_pcap():
    if 'file' not in request.files:
        return jsonify(ok=False, error="Aucun fichier 'file'"), 400
    f = request.files['file']
    if f.filename == "":
        return jsonify(ok=False, error="Nom de fichier vide"), 400
    if not _allowed_file(f.filename):
        return jsonify(ok=False, error="Extension non autorisée (.pcap/.pcapng)"), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    fname = secure_filename(os.path.basename(f.filename))
    if not fname:
        return jsonify(ok=False, error="Nom de fichier invalide"), 400
    pcap_path = os.path.join(UPLOAD_DIR, fname)
    try:
        f.save(pcap_path)
    except Exception as e:
        return jsonify(ok=False, error=f"Impossible d'enregistrer le fichier: {repr(e)}"), 500

    # MLflow contexte
    t0 = time.time()
    artifacts_to_log = []
    run_params = {
        "pcap_file": fname,
        "model_path": MODEL_PATH,
        "feature_list_path": FEATURE_LIST_PATH,
        "best_threshold": float(best_threshold),
        "iforest_loaded": bool(iforest_pack is not None),
        "ae_loaded": bool(ae_pack is not None),
        "syn_thresh_global": THRESH_SYN_GLOBAL,
    }

    if (pipe is None) or (feature_list is None):
        mlflow_log_inference(f"infer_error:{fname}", {**run_params, "error_stage": "pipeline_missing"},
                             {"duration_s": time.time()-t0, "n_flows": 0, "alert_rate": 0.0}, [])
        return jsonify(ok=False, error="Pipeline non chargé."), 500
    if not os.path.exists(TSHARK_EXE):
        mlflow_log_inference(f"infer_error:{fname}", {**run_params, "error_stage": "tshark_missing"},
                             {"duration_s": time.time()-t0, "n_flows": 0, "alert_rate": 0.0}, [])
        return jsonify(ok=False, error=f"TShark introuvable à: {TSHARK_EXE}"), 500

    # 1) tshark -> df
    tshark_fields = [
        "-T","fields",
        "-e","frame.time_epoch","-e","ip.src","-e","ip.dst","-e","ip.proto",
        "-e","frame.len","-e","tcp.srcport","-e","tcp.dstport",
        "-e","udp.srcport","-e","udp.dstport","-e","ip.ttl",
        "-E","separator=\t"
    ]
    cmd = [TSHARK_EXE, "-r", pcap_path] + tshark_fields
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TSHARK_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        mlflow_log_inference(f"infer_error:{fname}", {**run_params, "error_stage": "tshark_timeout"},
                             {"duration_s": time.time()-t0, "n_flows": 0, "alert_rate": 0.0}, [])
        return jsonify(ok=False, error=f"tshark a expiré (> {TSHARK_TIMEOUT_SEC}s)."), 500
    except Exception as e:
        mlflow_log_inference(f"infer_error:{fname}", {**run_params, "error_stage": "tshark_exec", "exec_error": repr(e)},
                             {"duration_s": time.time()-t0, "n_flows": 0, "alert_rate": 0.0}, [])
        return jsonify(ok=False, error=f"Erreur d'exécution tshark: {repr(e)}"), 500

    if res.returncode != 0:
        raw_out_err = os.path.join(UPLOAD_DIR, os.path.splitext(fname)[0] + "_raw_tshark.txt")
        _save_text(raw_out_err, "=== STDOUT ===\n" + (res.stdout or "") + "\n\n=== STDERR ===\n" + (res.stderr or ""))
        artifacts_to_log.append(raw_out_err)
        mlflow_log_inference(f"infer_error:{fname}", {**run_params, "error_stage": "tshark_nonzero"},
                             {"duration_s": time.time()-t0, "n_flows": 0, "alert_rate": 0.0}, artifacts_to_log)
        return jsonify(ok=False, error=f"tshark a échoué: {res.stderr[:300]}"), 500

    lines = [l for l in (res.stdout or "").strip().split("\n") if l.strip()]
    rows  = [l.split("\t") for l in lines]
    cols  = ["time","src","dst","proto","length","tcp_sport","tcp_dport","udp_sport","udp_dport","ip_ttl"]
    df    = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    raw_out = os.path.join(UPLOAD_DIR, os.path.splitext(fname)[0] + "_raw_tshark.txt")
    _save_text(raw_out, "=== STDOUT ===\n" + (res.stdout or "") + "\n\n=== STDERR ===\n" + (res.stderr or ""))
    artifacts_to_log.append(raw_out)

    # 2) Nettoyage (types) + signatures DF **AVANT** le drop des colonnes ports
    if not df.empty:
        df = df[df["proto"].apply(lambda x: str(x).isdigit())].copy()
        df["time"]   = pd.to_numeric(df["time"],   errors="coerce").fillna(0.0)
        df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(0.0)
        df["proto"]  = pd.to_numeric(df["proto"],  errors="coerce").fillna(-1).astype(int)
        df["ip_ttl"] = pd.to_numeric(df["ip_ttl"], errors="coerce").fillna(0).astype(int)
    alerts, sig_agg = detect_signatures_from_df(df)

    # ports fusionnés puis drop des colonnes brutes
    if not df.empty:
        df["sport"] = np.where(df["tcp_sport"]!="", df["tcp_sport"], df["udp_sport"])
        df["dport"] = np.where(df["tcp_dport"]!="", df["tcp_dport"], df["udp_dport"])
        df.drop(columns=["tcp_sport","tcp_dport","udp_sport","udp_dport"], inplace=True, errors="ignore")
        df["sport"] = pd.to_numeric(df["sport"], errors="coerce").fillna(0).astype(int)
        df["dport"] = pd.to_numeric(df["dport"], errors="coerce").fillna(0).astype(int)

    if df.empty:
        # push UI si alertes signatures
        if alerts:
            for i, (reason, dst, dport) in enumerate(alerts, start=1):
                socketio.emit('ids_update', {
                    "id": i, "status": "ATTACK", "timestamp": time.time(),
                    "feature_count": len(feature_list or []), "prob_attack": 1.0,
                    "src": "multiple", "dst": dst, "proto": "tcp",
                    "sport": 0, "dport": int(dport or 0)
                }, namespace="/ids")
        mlflow_log_inference(
            f"infer:{fname}",
            {**run_params, "early_return": "no_ip_packets", "alerts": len(alerts)},
            {"duration_s": time.time()-t0, "n_flows": 0, "alert_rate": 0.0},
            artifacts_to_log
        )
        return jsonify(ok=True, n_flows=0, results=[], info="Aucun paquet IP exploitable.")

    # 3) Agrégation flows
    flows = df.groupby(["src","dst","proto","sport","dport"]).agg(
        dur=("time", lambda x: x.max() - x.min()),
        spkts=("time", "count"),
        sbytes=("length", "sum")
    ).reset_index()

    if flows.empty:
        mlflow_log_inference(
            f"infer:{fname}",
            {**run_params, "early_return": "no_flows_after_groupby"},
            {"duration_s": time.time()-t0, "n_flows": 0, "alert_rate": 0.0},
            artifacts_to_log
        )
        socketio.emit('status_control', {'state':'IDLE','message':'Aucun flow après agrégation.'}, namespace='/ids')
        return jsonify(ok=True, n_flows=0, results=[], info="Aucun flow après agrégation.")

    per_dst = df.groupby(['dst','dport']).agg(
        dst_total_pkts=('time','count'),
        dst_total_bytes=('length','sum'),
        dst_unique_src=('src', lambda x: x.nunique())
    ).reset_index()
    flows = flows.merge(per_dst, on=['dst','dport'], how='left')
    flows["dst_total_pkts"]  = flows["dst_total_pkts"].fillna(0).astype(int)
    flows["dst_total_bytes"] = flows["dst_total_bytes"].fillna(0).astype(float)
    flows["dst_unique_src"]  = flows["dst_unique_src"].fillna(0).astype(int)

    # dérivées
    flows["dpkts"] = flows["spkts"]
    flows["dbytes"] = flows["sbytes"]
    flows["rate"]   = flows["sbytes"] / (flows["dur"] + 1e-6)
    flows["service"], flows["state"] = "unknown", "CON"

    # TTL moyen
    ttl_agg = df.groupby(["src","dst","proto","sport","dport"]).agg(sttl=("ip_ttl","mean")).reset_index()
    flows = flows.merge(ttl_agg, on=["src","dst","proto","sport","dport"], how="left")
    flows["sttl"] = flows["sttl"].fillna(0.0)
    flows["smean"] = flows["sbytes"] / (flows["spkts"] + 1e-6)

    # map proto -> string
    flows["proto"] = flows["proto"].map(lambda v: PROTO_MAP.get(int(v), "other")).astype(str)

    # signatures déjà calculées
    flows["total_syn_count"] = float(sig_agg.get("total_syn", 0))
    flows["malformed_ratio"] = float(sig_agg.get("malformed_ratio", 0.0))
    flows["syn_count_dst"]   = 0  # (rapide: pas de détail par dst dans cette version)

    preflows_csv = os.path.join(UPLOAD_DIR, os.path.splitext(fname)[0] + "_flows_pre.csv")
    _save_text(preflows_csv, flows.to_csv(index=False))
    artifacts_to_log.append(preflows_csv)

    # 4) Construire X pour le modèle
    X = pd.DataFrame(index=flows.index)
    for feat in (feature_list or []):
        if feat in flows.columns:
            X[feat] = flows[feat]
        else:
            X[feat] = "unknown" if feat in ("proto","service","state") else 0.0

    for col in X.columns:
        if col not in ("proto","service","state"):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)
        else:
            X[col] = X[col].astype(str)

    xhead_csv = os.path.join(UPLOAD_DIR, os.path.splitext(fname)[0] + "_X_head.csv")
    _save_text(xhead_csv, X.reset_index(drop=True).head(50).to_csv(index=False))
    artifacts_to_log.append(xhead_csv)

    if X.empty:
        mlflow_log_inference(
            f"infer:{fname}",
            {**run_params, "early_return": "empty_feature_matrix"},
            {"duration_s": time.time()-t0, "n_flows": 0, "alert_rate": 0.0},
            artifacts_to_log
        )
        socketio.emit('status_control', {'state':'IDLE','message':'Aucun flow exploitable pour ML.'}, namespace='/ids')
        return jsonify(ok=True, n_flows=0, results=[], info="Aucun flow exploitable pour ML.")

    # 5) prédiction XGB
    try:
        probs = pipe.predict_proba(X)[:, 1]
    except Exception as e:
        log(f"DEBUG: predict_proba failed: {e}")
        try:
            preds = pipe.predict(X)
            probs = np.where(preds==1, max(0.75, best_threshold), min(0.25, best_threshold))
        except Exception as e2:
            log(f"❌ Échec prédiction ML: {e2}")
            mlflow_log_inference(
                f"infer_error:{fname}",
                {**run_params, "error_stage": "predict"},
                {"duration_s": time.time()-t0, "n_flows": int(len(flows)), "alert_rate": 0.0},
                artifacts_to_log
            )
            socketio.emit('status_control', {'state':'IDLE','message':'Erreur prédiction ML.'}, namespace='/ids')
            return jsonify(ok=True, n_flows=0, results=[], info="Erreur prédiction ML.")
    flows["prob_attack_xgb"] = probs

    # 6) Scoring IForest
    if iforest_pack is not None:
        try:
            X_if = flows[num_cols_if].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            scores_if = -iforest_pack.named_steps["if"].score_samples(
                iforest_pack.named_steps["rs"].transform(X_if)
            )
            flows["anomaly_score"] = scores_if
        except Exception as e:
            log(f"IForest scoring failed: {e}")
            flows["anomaly_score"] = 0.0
    else:
        flows["anomaly_score"] = 0.0

    # 7) Scoring AE
    if ae_pack is not None:
        try:
            X_ae = flows[num_cols_ae].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
            scaler = ae_pack["scaler"]; mlp = ae_pack["mlp"]
            Xs = scaler.transform(X_ae); Xhat = mlp.predict(Xs)
            flows["ae_mse"] = np.mean((Xhat - Xs)**2, axis=1)
        except Exception as e:
            log(f"AE scoring failed: {e}")
            flows["ae_mse"] = 0.0
    else:
        flows["ae_mse"] = 0.0

    # 8) Ensemble
    EPS = 1e-12
    alpha, beta, gamma = 0.7, 0.2, 0.1

    def _norm_series(series, tau):
        if tau is None or tau <= 0: return np.zeros_like(series, dtype=float)
        z = (series - tau) / (tau + 1e-6)
        z = np.clip(z, 0.0, 1.0)
        return z

    anom_if01 = _norm_series(flows["anomaly_score"], tau_iforest)
    ae01      = _norm_series(flows["ae_mse"],        tau_mse)
    flows["ensemble_prob"] = alpha*flows["prob_attack_xgb"] + beta*anom_if01 + gamma*ae01
    flows["pred"] = (flows["ensemble_prob"] + EPS >= best_threshold).astype(int)
    flows["status"] = np.where(flows["pred"]==1, "ATTACK", "NORMAL")

    # 9) Sauvegarde + push UI
    out_csv = os.path.join(UPLOAD_DIR, os.path.splitext(fname)[0] + "_with_preds.csv")
    _save_text(out_csv, flows.to_csv(index=False))
    artifacts_to_log.append(out_csv)

    for i, row in flows.iterrows():
        socketio.emit('ids_update', {
            "id": int(i)+1,
            "status": row["status"],
            "timestamp": time.time(),
            "feature_count": len(feature_list or []),
            "prob_attack": float(row["ensemble_prob"]),
            "prob_xgb": float(row["prob_attack_xgb"]),
            "anom_score": float(row["anomaly_score"]),
            "ae_mse": float(row["ae_mse"]),
            "src": row["src"], "dst": row["dst"], "proto": row["proto"],
            "sport": int(row["sport"]), "dport": int(row["dport"])
        }, namespace="/ids")

    duration = time.time() - t0
    alert_rate = float(np.mean(flows["pred"])) if len(flows) else 0.0
    run_metrics = {
        "duration_s": duration,
        "n_flows": int(len(flows)),
        "alert_rate": alert_rate,
        "mean_prob_xgb": float(np.mean(flows["prob_attack_xgb"])) if "prob_attack_xgb" in flows else 0.0,
        "mean_prob_ensemble": float(np.mean(flows["ensemble_prob"])) if "ensemble_prob" in flows else 0.0,
    }
    mlflow_log_inference(
        run_name=f"infer:{fname}",
        params=run_params,
        metrics=run_metrics,
        artifacts=artifacts_to_log
    )

    socketio.emit('status_control', {'state':'IDLE','message':'Analyse PCAP terminée.'}, namespace='/ids')
    preview = flows[["src","dst","proto","sport","dport","status","ensemble_prob","prob_attack_xgb","anomaly_score","ae_mse"]].head(20).to_dict(orient="records")
    return jsonify(ok=True, n_flows=int(len(flows)), csv_path=out_csv, preview=preview)

# ============== MAIN ================================
if __name__ == '__main__':
    log("Starting Flask-SocketIO Server on http://0.0.0.0:5000")
    # Important: pas de double init. socketio.run prend app déjà initialisée.
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
# ci: trigger rebuild Sun Oct 26 23:04:03 WAT 2025
# ci: trigger rebuild Mon Oct 27 01:12:33 WAT 2025
