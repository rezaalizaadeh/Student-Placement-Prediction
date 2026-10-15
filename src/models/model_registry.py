"""Model Registry Demo using MLflow"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


EXPERIMENT_NAME = "Student Placement Prediction"
MODEL_NAME = "student-placement-model"


def get_project_root():
    return Path(__file__).resolve().parents[2]


def configure_mlflow():
    project_root = get_project_root()
    db_path = str(project_root / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")


def get_latest_run_id():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run training first."
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in experiment.")

    return runs[0].info.run_id


def register_model():
    configure_mlflow()

    run_id = get_latest_run_id()
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"Model registered -> version {result.version}")


def list_models():
    configure_mlflow()

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        print("No versions found.")
        return

    for v in versions:
        print(f"Version {v.version} | Run {v.run_id}")


def set_alias(alias, version):
    configure_mlflow()

    client = MlflowClient()
    client.set_registered_model_alias(
        MODEL_NAME,
        alias,
        version
    )

    print(f"Alias '{alias}' -> version {version}")


def load_model(alias=None, version=None):
    configure_mlflow()

    if version:
        uri = f"models:/{MODEL_NAME}/{version}"
    else:
        uri = f"models:/{MODEL_NAME}@{alias}"

    model = mlflow.sklearn.load_model(uri)

    print(f"Loaded model from {uri}")
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["register", "list", "set-alias", "load"]
    )
    parser.add_argument("--alias")
    parser.add_argument("--version")

    args = parser.parse_args()

    if args.command == "register":
        register_model()

    elif args.command == "list":
        list_models()

    elif args.command == "set-alias":
        if not args.alias or not args.version:
            raise ValueError("Provide --alias and --version")

        set_alias(args.alias, args.version)

    elif args.command == "load":
        load_model(args.alias, args.version)


if __name__ == "__main__":
    main()