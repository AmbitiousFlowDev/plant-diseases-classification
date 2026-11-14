import mlflow
from pathlib import Path

def setup(project_root: Path, experiment: str):
    """Local SQLite store (project_root/mlflow.db), artifacts in project_root/mlruns/<experiment>."""
    mlflow.set_tracking_uri(f"sqlite:///{Path(project_root) / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment,
            artifact_location=(Path(project_root) / "mlruns" / experiment).as_uri())
    mlflow.set_experiment(experiment)

def flatten(d: dict, prefix: str = "") -> dict:
    """{'split': {'train': .7}} -> {'split_train': .7}  (MLflow params must be flat)"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, prefix=f"{key}_"))
        else:
            out[key] = v
    return out