"""Preprocessing entrypoint for the Airflow DAG."""

from __future__ import annotations

from pathlib import Path


def preprocess_data() -> str:
    """Prepare a cleaned raw dataset file used by downstream tasks."""
    import pandas as pd

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    preferred_source = raw_dir / "student_placement_data_v1.csv"
    fallback_source = raw_dir / "student_placement_data.csv"
    source = preferred_source if preferred_source.exists() else fallback_source

    if not source.exists():
        raise FileNotFoundError(
            "No input dataset found in data/raw/. Expected "
            "student_placement_data_v1.csv or student_placement_data.csv."
        )

    frame = pd.read_csv(source)
    frame = frame.drop_duplicates().reset_index(drop=True)

    output_path = raw_dir / "student_placement_data.csv"
    frame.to_csv(output_path, index=False)

    return str(output_path)