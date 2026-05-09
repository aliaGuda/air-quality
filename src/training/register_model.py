import json
import os
import time

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_best_run(path: str = "models/best_run.json") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def wait_for_model_version(client, model_name: str, version: str) -> None:
    for _ in range(30):
        model_version = client.get_model_version(model_name, version)
        if model_version.status == "READY":
            return
        time.sleep(1)

    raise TimeoutError(f"Model version {version} was not ready in time.")


def validate_run_has_model_artifact(client, run_id: str) -> None:
    root_artifacts = client.list_artifacts(run_id)
    root_paths = [artifact.path for artifact in root_artifacts]

    if "model" not in root_paths:
        raise RuntimeError(
            f"No 'model' artifact found for run {run_id}. "
            f"Available root artifacts: {root_paths}. "
            "Run train.py again and confirm the model artifact is visible in MLflow."
        )

    model_artifacts = client.list_artifacts(run_id, "model")
    model_paths = [artifact.path for artifact in model_artifacts]

    if not model_paths:
        raise RuntimeError(
            f"The 'model' artifact folder exists for run {run_id}, "
            "but it appears empty. Do not register this run."
        )

    print(f"Validated model artifact for run {run_id}.")
    print(f"Model artifact contents: {model_paths}")


def main() -> None:
    params = load_params()
    best_run = load_best_run()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        params["mlflow"]["tracking_uri"],
    )

    registered_model_name = params["training"]["model_name"]
    run_id = best_run["run_id"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Best run ID: {run_id}")
    print(f"Registered model name: {registered_model_name}")

    validate_run_has_model_artifact(client, run_id)

    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    version = registered_model.version

    wait_for_model_version(client, registered_model_name, version)

    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print(f"Model {registered_model_name} version {version} moved to Staging.")

    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"Model {registered_model_name} version {version} moved to Production.")


if __name__ == "__main__":
    main()