import os
import shutil
from datetime import datetime, timezone

import joblib

from services.database_service import get_active_deployment, get_model_result, record_deployment
from services.model_registry import MODEL_NAMES


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ACTIVE_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "deployed_model.joblib")
ACTIVE_MODEL_PATH = os.path.abspath(
    os.environ.get("ALGOGUARD_DEPLOYED_MODEL_PATH", DEFAULT_ACTIVE_MODEL_PATH)
)


class DeploymentError(RuntimeError):
    pass


def deploy_model(model_id, admin_id):
    """Atomically promote one completed recommendation to active deployment."""
    model = get_model_result(model_id)
    if not model:
        raise DeploymentError("The selected model result does not exist.")
    if model.get("model_name") not in MODEL_NAMES:
        raise DeploymentError(
            "This legacy model is no longer supported. Train a new run before deploying."
        )
    if model.get("evaluation_status") != "completed":
        raise DeploymentError("Only a completed model evaluation can be deployed.")
    if not model.get("is_recommended"):
        raise DeploymentError("Only the recommended winner for a run can be deployed.")

    source_path = model.get("artifact_path")
    if not source_path or not os.path.exists(source_path):
        raise DeploymentError("The selected model artifact is missing.")

    try:
        artifact = joblib.load(source_path)
    except Exception as error:
        raise DeploymentError(
            f"The selected model artifact could not be loaded: {error}"
        ) from error

    metric_summary = {
        "accuracy": model.get("accuracy"),
        "precision": model.get("precision_score"),
        "recall": model.get("recall"),
        "f1_score": model.get("f1_score"),
        "roc_auc": model.get("roc_auc"),
        "false_positive_rate": model.get("fpr"),
        "cpu_time_seconds": model.get("cpu_usage"),
        "peak_ram_increase_mb": model.get("ram_usage"),
        "model_size_mb": model.get("model_size"),
        "overall_score": model.get("overall_score"),
    }
    artifact["database_model_id"] = model["model_id"]
    artifact["source_run_id"] = model["run_id"]
    artifact["model_name"] = model["model_name"]
    artifact["model_type"] = model["model_type"]
    artifact["metric_summary"] = metric_summary
    artifact["deployment_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(ACTIVE_MODEL_PATH), exist_ok=True)
    temporary_path = f"{ACTIVE_MODEL_PATH}.tmp"
    backup_path = f"{ACTIVE_MODEL_PATH}.backup"

    try:
        joblib.dump(artifact, temporary_path)
        joblib.load(temporary_path)
    except Exception as error:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise DeploymentError(f"Deployment artifact validation failed: {error}") from error

    had_active_file = os.path.exists(ACTIVE_MODEL_PATH)
    if had_active_file:
        shutil.copy2(ACTIVE_MODEL_PATH, backup_path)

    try:
        os.replace(temporary_path, ACTIVE_MODEL_PATH)
        deployment = record_deployment(
            model["model_id"],
            model["run_id"],
            admin_id,
            ACTIVE_MODEL_PATH,
        )
    except Exception as error:
        if os.path.exists(backup_path):
            os.replace(backup_path, ACTIVE_MODEL_PATH)
        elif not had_active_file and os.path.exists(ACTIVE_MODEL_PATH):
            os.remove(ACTIVE_MODEL_PATH)
        raise DeploymentError(f"Model deployment failed: {error}") from error
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        if os.path.exists(backup_path):
            os.remove(backup_path)

    return {"model": model, "deployment": deployment, "artifact_path": ACTIVE_MODEL_PATH}


def load_active_artifact():
    """Load only the model referenced by the active deployment record."""
    deployment = get_active_deployment()
    if not deployment:
        raise DeploymentError("No model has been deployed. Deploy a recommended model first.")
    if deployment.get("model_name") not in MODEL_NAMES:
        raise DeploymentError(
            "The active deployment uses a removed model. Deploy a model from a new run."
        )

    artifact_path = deployment.get("artifact_path") or ACTIVE_MODEL_PATH
    if not os.path.exists(artifact_path):
        raise DeploymentError("The active deployed model artifact is missing.")

    try:
        artifact = joblib.load(artifact_path)
    except Exception as error:
        raise DeploymentError(f"The active model could not be loaded: {error}") from error

    if artifact.get("database_model_id") != deployment.get("model_id"):
        raise DeploymentError("The active artifact does not match the deployment record.")
    return artifact, deployment
