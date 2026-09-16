"""Evaluation entrypoint for the Airflow DAG."""

from __future__ import annotations

from pathlib import Path


def evaluate() -> dict:
    """Evaluate the saved model and persist evaluation metrics."""
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "student_placement_data.csv"
    model_path = project_root / "models" / "model.pkl"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run preprocessing first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first."
        )

    frame = pd.read_csv(data_path)
    if "Placement" not in frame.columns:
        raise ValueError("Expected target column 'Placement' in dataset.")

    features = frame.drop("Placement", axis=1)
    target = frame["Placement"]

    _, x_test, _, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = joblib.load(model_path)
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "evaluation_metrics.json"
    pd.Series(metrics).to_json(metrics_path)

    return metrics
