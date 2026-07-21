import io
import json
import sqlite3

import joblib
import pytest

from services import database_service as db
from services.deployment_service import (
    STACKING_QUALITY_THRESHOLDS,
    DeploymentError,
    deploy_model,
    stacking_deployment_eligibility,
)
from services.model_registry import MODEL_IDS, MODEL_WORKFLOW_VERSION
from services.simulation_service import run_simulation


def test_database_context_manager_closes_connection():
    with db.get_connection() as connection:
        connection.execute("SELECT 1").fetchone()

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def synthetic_results():
    results = []
    for index, (model_id, model_name) in enumerate(MODEL_IDS.items(), start=1):
        value = 90.0 - index
        normalized = {
            "accuracy": value / 100,
            "precision": value / 100,
            "recall": value / 100,
            "f1_score": value / 100,
            "roc_auc": value / 100,
            "false_positive_rate": value / 100,
            "cpu_usage": value / 100,
            "ram_usage": value / 100,
            "model_size": value / 100,
        }
        comparison_metrics = normalized if index <= 5 else None
        results.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "model_type": "individual" if index <= 5 else "ensemble",
                "status": "completed",
                "accuracy": value,
                "precision": value,
                "recall": value,
                "f1_score": value,
                "roc_auc": value,
                "false_positive_rate": 100 - value,
                "cpu_usage": index / 10,
                "ram_usage": float(index),
                "model_size": index / 20,
                "processing_time": index / 10,
                "predicted_normal_count": 10,
                "predicted_anomaly_count": 10,
                "normalized_metrics": comparison_metrics,
                "overall_score": (
                    sum(comparison_metrics.values()) / 9 if comparison_metrics else None
                ),
                "rank": index if index <= 5 else None,
                "model_path": None,
            }
        )
    return {"model_results": results, "best_model": results[0]}


def post_csv(client, frame, filename):
    payload = frame.to_csv(index=False).encode("utf-8")
    return client.post(
        "/upload",
        data={"dataset": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


def model_id_for(run_id, model_name):
    with db.get_connection() as connection:
        row = connection.execute(
            "SELECT model_id FROM detection_model WHERE run_id = ? AND model_name = ?",
            (run_id, model_name),
        ).fetchone()
        assert row is not None
        return row["model_id"]


def test_each_csv_upload_creates_one_independent_run(
    authenticated_client,
    app_module,
    binary_dataframe,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module, "train_and_compare_models", lambda *args, **kwargs: synthetic_results()
    )
    before = db.list_training_runs()["total"]

    first = post_csv(authenticated_client, binary_dataframe.iloc[:64], "first.csv")
    second = post_csv(authenticated_client, binary_dataframe.iloc[32:], "second.csv")

    assert first.status_code == 302
    assert second.status_code == 302
    history = db.list_training_runs()
    assert history["total"] == before + 2
    newest, previous = history["items"][:2]
    assert newest["filename"] == "second.csv"
    assert previous["filename"] == "first.csv"
    assert newest["row_count"] == len(binary_dataframe.iloc[32:])
    assert previous["row_count"] == len(binary_dataframe.iloc[:64])
    assert newest["model_result_count"] == 6
    assert previous["model_result_count"] == 6


def test_one_class_upload_fails_safely(authenticated_client, binary_dataframe):
    one_class = binary_dataframe[binary_dataframe["label"] == "Normal"]
    response = post_csv(authenticated_client, one_class, "one_class.csv")
    assert response.status_code == 302
    run = db.get_latest_training_run()
    assert run["validation_status"] == "failed"
    assert run["training_status"] == "failed"
    assert "exactly two" in run["error_message"]


def test_top_individual_cannot_be_deployed(
    authenticated_client,
    trained_bundle,
):
    best_name = trained_bundle["result"]["best_model"]["model_name"]
    model_id = model_id_for(trained_bundle["run_id"], best_name)

    with pytest.raises(DeploymentError, match="Only the evaluated Stacking Ensemble"):
        deploy_model(model_id, 1)

    response = authenticated_client.post(f"/runs/{trained_bundle['run_id']}/deploy/{model_id}")
    assert response.status_code == 302


def test_eligible_stacking_can_be_deployed(authenticated_client, trained_bundle):
    model_id = model_id_for(trained_bundle["run_id"], "Stacking Ensemble")
    response = authenticated_client.post(f"/runs/{trained_bundle['run_id']}/deploy/{model_id}")
    assert response.status_code == 302
    active = db.get_active_deployment()
    assert active["model_id"] == model_id
    assert active["model_name"] == "Stacking Ensemble"


def test_failed_activation_restores_previous_artifact(trained_bundle, tmp_path, monkeypatch):
    from services import deployment_service

    active_path = tmp_path / "deployed_model.joblib"
    previous_contents = b"previous-active-artifact"
    active_path.write_bytes(previous_contents)
    model_id = model_id_for(trained_bundle["run_id"], "Stacking Ensemble")

    monkeypatch.setattr(deployment_service, "ACTIVE_MODEL_PATH", str(active_path))

    def fail_record_deployment(*_args, **_kwargs):
        raise RuntimeError("database activation failed")

    monkeypatch.setattr(deployment_service, "record_deployment", fail_record_deployment)

    with pytest.raises(DeploymentError, match="database activation failed"):
        deployment_service.deploy_model(model_id, 1)

    assert active_path.read_bytes() == previous_contents
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.backup")) == []


def test_active_artifact_with_invalid_format_fails_safely(tmp_path, monkeypatch):
    from services import deployment_service

    artifact_path = tmp_path / "invalid-active.joblib"
    joblib.dump(["not", "an", "artifact"], artifact_path)
    monkeypatch.setattr(
        deployment_service,
        "get_active_deployment",
        lambda: {
            "model_id": 999,
            "model_name": "Stacking Ensemble",
            "artifact_path": str(artifact_path),
        },
    )

    with pytest.raises(DeploymentError, match="invalid format"):
        deployment_service.load_active_artifact()


def test_deployed_stacking_can_predict(trained_bundle):
    model_id = model_id_for(trained_bundle["run_id"], "Stacking Ensemble")
    deploy_model(model_id, 1)
    result = run_simulation(trained_bundle["prepared"].feature_defaults)
    assert result["prediction"] in {"Normal", "Attack"}
    assert 0 <= result["confidence"] <= 100
    assert result["deployed_model_name"] == "Stacking Ensemble"


def test_stacking_quality_gate_blocks_low_results(monkeypatch):
    monkeypatch.setitem(STACKING_QUALITY_THRESHOLDS, "accuracy", 70.0)
    eligible, reason = stacking_deployment_eligibility(
        {
            "model_name": "Stacking Ensemble",
            "version": MODEL_WORKFLOW_VERSION,
            "run_training_status": "completed",
            "evaluation_status": "completed",
            "accuracy": 69.0,
            "f1_score": 80.0,
            "roc_auc": 80.0,
        }
    )
    assert eligible is False
    assert "Accuracy 69.00% is below 70.00%" in reason


def test_stacking_quality_gate_blocks_legacy_workflow():
    eligible, reason = stacking_deployment_eligibility(
        {
            "model_name": "Stacking Ensemble",
            "version": "legacy-workflow",
            "evaluation_status": "completed",
            "accuracy": 100.0,
            "f1_score": 100.0,
            "roc_auc": 100.0,
        }
    )
    assert eligible is False
    assert "legacy workflow" in reason


def test_stacking_quality_gate_blocks_failed_training_run():
    eligible, reason = stacking_deployment_eligibility(
        {
            "model_name": "Stacking Ensemble",
            "version": MODEL_WORKFLOW_VERSION,
            "run_training_status": "failed",
            "evaluation_status": "completed",
            "accuracy": 100.0,
            "f1_score": 100.0,
            "roc_auc": 100.0,
        }
    )
    assert eligible is False
    assert "training run must complete successfully" in reason


def test_results_ui_separates_comparison_from_stacking_deployment(
    authenticated_client,
    app_module,
    trained_bundle,
    monkeypatch,
):
    monkeypatch.setattr(app_module, "get_active_deployment", lambda: None)
    response = authenticated_client.get(f"/runs/{trained_bundle['run_id']}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Top Individual" in html
    assert "Final Deployment Framework" in html
    assert "Quality Gate Passed" in html
    assert "Deploy Stacking" in html
    assert "Deploy Best Model" not in html
    assert "Stacking Ensemble" in html
    assert "Not ranked" in html


def payload_for_prediction(trained_bundle, expected_class):
    model_row = next(
        result
        for result in trained_bundle["result"]["model_results"]
        if result["model_name"] == "Stacking Ensemble"
    )
    pipeline = joblib.load(model_row["model_path"])["pipeline"]
    features = trained_bundle["frame"].iloc[:, :-1]
    predictions = pipeline.predict(features)
    index = next(index for index, value in enumerate(predictions) if int(value) == expected_class)
    return json.loads(features.iloc[index].to_json())


def deploy_stacking(trained_bundle):
    model_id = model_id_for(trained_bundle["run_id"], "Stacking Ensemble")
    deploy_model(model_id, 1)


def test_attack_prediction_creates_alert_and_logs(authenticated_client, trained_bundle):
    deploy_stacking(trained_bundle)
    before_alerts = len(db.list_alerts(limit=1000))
    response = authenticated_client.post(
        "/predict",
        json=payload_for_prediction(trained_bundle, 1),
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["prediction"] == "Attack"
    assert payload["alert_created"] is True
    assert len(db.list_alerts(limit=1000)) == before_alerts + 1
    logs = db.list_system_logs({"search": "classified as Attack"})
    assert logs["total"] >= 1


def test_normal_prediction_does_not_create_alert(authenticated_client, trained_bundle):
    deploy_stacking(trained_bundle)
    before_alerts = len(db.list_alerts(limit=1000))
    response = authenticated_client.post(
        "/predict",
        json=payload_for_prediction(trained_bundle, 0),
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["prediction"] == "Normal"
    assert payload["alert_created"] is False
    assert len(db.list_alerts(limit=1000)) == before_alerts


def test_system_logs_ui_loads_and_filters(authenticated_client, trained_bundle):
    db.log_system_event(
        1,
        "QA Unique Module",
        "qa_filter_event",
        "Success",
        "127.0.0.1",
        message="Unique searchable audit event.",
        run_id=trained_bundle["run_id"],
        model_name="Random Forest",
    )
    response = authenticated_client.get(
        "/logs?module=QA+Unique+Module&status=Success&model=Random+Forest&search=Unique+searchable"
    )
    assert response.status_code == 200
    assert b"Unique searchable audit event" in response.data
    assert b"System Logs" in response.data


def test_required_log_categories_are_recorded(authenticated_client, trained_bundle):
    model_id = model_id_for(trained_bundle["run_id"], "Stacking Ensemble")
    authenticated_client.post(f"/runs/{trained_bundle['run_id']}/deploy/{model_id}")
    modules = {row["module"] for row in db.list_system_logs(page=1, per_page=100)["items"]}
    assert {"Authentication", "Dataset", "Training", "Deployment"}.issubset(modules)


def test_authenticated_flask_pages_smoke(authenticated_client, trained_bundle):
    paths = [
        "/",
        "/upload",
        "/history",
        f"/runs/{trained_bundle['run_id']}",
        "/simulation",
        "/alerts",
        "/logs",
        "/admins",
    ]
    assert all(authenticated_client.get(path).status_code == 200 for path in paths)
