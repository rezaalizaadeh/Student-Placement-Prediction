from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_placement_data_v1.csv"
)

CURRENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_data.csv"
)

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "train_model.py"
)

DRIFT_THRESHOLD = 1.5


# Load reference and current data
reference_data = pd.read_csv(REFERENCE_PATH)
current_data = pd.read_csv(CURRENT_PATH)


# Simple drift rule used in the course demo
reference_mean = reference_data["CGPA"].mean()
current_mean = current_data["CGPA"].mean()

cgpa_mean_difference = abs(
    reference_mean - current_mean
)


print(f"Reference CGPA mean: {reference_mean:.4f}")
print(f"Current CGPA mean: {current_mean:.4f}")
print(
    f"CGPA mean difference: "
    f"{cgpa_mean_difference:.4f}"
)


# Trigger retraining when drift exceeds threshold
if cgpa_mean_difference > DRIFT_THRESHOLD:
    print("Significant drift detected.")
    print("Starting automatic model retraining...")

    subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        check=True,
        cwd=PROJECT_ROOT,
    )

    print("Retraining completed successfully.")

else:
    print("No major drift detected.")
    print("Retraining is not required.")