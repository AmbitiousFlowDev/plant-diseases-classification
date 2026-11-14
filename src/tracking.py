import sqlite3
from pathlib import Path

import mlflow


def setup(project_root: Path, experiment: str):
    """Local SQLite store (project_root/mlflow.db), artifacts in project_root/mlruns/<experiment>."""
    project_root = Path(project_root)
    db = project_root / "mlflow.db"
    local_artifacts = (project_root / "mlruns" / experiment).as_uri()

    mlflow.set_tracking_uri(f"sqlite:///{db}")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        mlflow.create_experiment(experiment, artifact_location=local_artifacts)
    elif exp.artifact_location != local_artifacts:
        # mlflow.db is shared through git, so the stored location can be another machine's
        # absolute path. MLflow has no API to change it; point new runs at this checkout.
        with sqlite3.connect(db) as con:
            con.execute("UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                        (local_artifacts, int(exp.experiment_id)))
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
