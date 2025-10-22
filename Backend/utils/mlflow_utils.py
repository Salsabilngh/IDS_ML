# utils/mlflow_utils.py
import os

# --- Essayez d'importer mlflow ; sinon, stubs no-op pour ne pas casser les scripts ---
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    _MLFLOW_AVAILABLE = False


def _default_tracking_uri() -> str:
    """
    Retourne un tracking URI local sous le dossier du projet (../mlruns).
    """
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_dir = os.path.join(base, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    # format file:///C:/... pour Windows
    return f"file:///{mlruns_dir.replace(os.sep, '/')}"


def setup_mlflow(experiment_name: str = "Default", tracking_uri: str | None = None) -> None:
    """
    Configure MLflow : tracking URI + expérience.
    Si mlflow n'est pas dispo, on affiche un message et on continue sans logging.
    """
    if not _MLFLOW_AVAILABLE:
        print("[mlflow_utils] MLflow non installé : logging désactivé.")
        return

    uri = tracking_uri or _default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    print(f"[mlflow_utils] Tracking URI = {uri}")
    print(f"[mlflow_utils] Experiment   = {experiment_name}")


class _DummyRun:
    def __enter__(self, *a, **k): return self
    def __exit__(self, exc_type, exc, tb): return False


def start_run(run_name: str | None = None):
    """
    Démarre un run MLflow si dispo, sinon renvoie un context manager no-op.
    """
    if not _MLFLOW_AVAILABLE:
        return _DummyRun()
    return mlflow.start_run(run_name=run_name)


def log_params(d: dict) -> None:
    if _MLFLOW_AVAILABLE and d:
        mlflow.log_params(d)


def log_metrics(d: dict) -> None:
    if _MLFLOW_AVAILABLE and d:
        mlflow.log_metrics(d)


def log_artifacts(paths, artifact_path: str | None = None) -> None:
    if not _MLFLOW_AVAILABLE:
        return
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    for p in paths:
        if p and os.path.exists(p):
            if artifact_path:
                mlflow.log_artifact(p, artifact_path=artifact_path)
            else:
                mlflow.log_artifact(p)

