from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "student_placement_data_v1.csv"
)

CURRENT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "current_data.csv"
)

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "placement_monitoring_report.html"
)


# Check input files
for data_path, label in [
    (REFERENCE_PATH, "reference"),
    (CURRENT_PATH, "current"),
]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {label} data file: {data_path}"
        )

    if data_path.stat().st_size == 0:
        raise ValueError(
            f"{label.title()} data file is empty: {data_path}"
        )


# Load datasets
reference_data = pd.read_csv(REFERENCE_PATH)
current_data = pd.read_csv(CURRENT_PATH)


# Check that both datasets have the same columns
if list(reference_data.columns) != list(current_data.columns):
    raise ValueError(
        "Reference and current datasets must have the same columns."
    )


# Create drift report
report = Report([
    DataDriftPreset()
])

snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data,
)


# Save HTML report
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

snapshot.save_html(str(REPORT_PATH))

print(f"Report generated successfully: {REPORT_PATH}")