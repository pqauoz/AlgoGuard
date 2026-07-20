import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


TEST_RUNTIME = Path(tempfile.mkdtemp(prefix="algoguard-tests-"))
os.environ["ALGOGUARD_DATABASE_PATH"] = str(TEST_RUNTIME / "test.sqlite3")
os.environ["ALGOGUARD_DEPLOYED_MODEL_PATH"] = str(TEST_RUNTIME / "models" / "deployed_model.joblib")
os.environ["ALGOGUARD_SECRET_KEY"] = "test-secret"


@pytest.fixture(scope="session")
def app_module(tmp_path_factory):
    import app as module

    runtime = tmp_path_factory.mktemp("flask-runtime")
    module.app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(runtime / "uploads"),
        SAVED_MODEL_FOLDER=str(runtime / "models"),
        REPORT_FOLDER=str(runtime / "reports"),
    )
    for key in ("UPLOAD_FOLDER", "SAVED_MODEL_FOLDER", "REPORT_FOLDER"):
        Path(module.app.config[key]).mkdir(parents=True, exist_ok=True)
    return module


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture()
def authenticated_client(client):
    response = client.post(
        "/login",
        data={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 302
    return client


@pytest.fixture(scope="session")
def binary_dataframe():
    rng = np.random.default_rng(42)
    rows_per_class = 48
    normal = pd.DataFrame(
        {
            "duration": rng.normal(1.0, 0.15, rows_per_class),
            "packet_rate": rng.normal(30, 4, rows_per_class),
            "source_bytes": rng.normal(500, 60, rows_per_class),
            "protocol": rng.choice(["tcp", "udp"], rows_per_class),
            "state": rng.choice(["CON", "FIN"], rows_per_class),
            "label": "Normal",
        }
    )
    attack = pd.DataFrame(
        {
            "duration": rng.normal(5.5, 0.25, rows_per_class),
            "packet_rate": rng.normal(900, 45, rows_per_class),
            "source_bytes": rng.normal(9000, 400, rows_per_class),
            "protocol": rng.choice(["tcp", "icmp"], rows_per_class),
            "state": rng.choice(["REQ", "INT"], rows_per_class),
            "label": "Attack",
        }
    )
    return (
        pd.concat([normal, attack], ignore_index=True)
        .sample(
            frac=1,
            random_state=42,
        )
        .reset_index(drop=True)
    )


@pytest.fixture(scope="session")
def prepared_dataset(binary_dataframe, tmp_path_factory):
    from services.preprocessing_service import prepare_dataset

    csv_path = tmp_path_factory.mktemp("fixtures") / "binary_traffic.csv"
    binary_dataframe.to_csv(csv_path, index=False)
    return prepare_dataset(csv_path)


@pytest.fixture(scope="session")
def trained_bundle(app_module, binary_dataframe, prepared_dataset, tmp_path_factory):
    from services import database_service as db
    from services.training_service import train_and_compare_models

    run_id = db.create_training_run(1, "binary_traffic.csv", "binary_traffic.csv")
    db.update_training_run(
        run_id,
        row_count=prepared_dataset.summary["rows"],
        feature_count=prepared_dataset.summary["features"],
        normal_count=prepared_dataset.summary["normal_count"],
        anomaly_count=prepared_dataset.summary["anomaly_count"],
        target_column=prepared_dataset.summary["target_column"],
        class_distribution=prepared_dataset.summary["label_counts"],
        validation_status="passed",
        preprocessing_status="completed",
        training_status="in_progress",
    )

    def event_logger(module, action, status, model_name=None, message=None):
        db.log_system_event(
            1,
            module,
            action,
            status,
            "127.0.0.1",
            message=message,
            run_id=run_id,
            dataset_filename="binary_traffic.csv",
            model_name=model_name,
        )

    model_folder = tmp_path_factory.mktemp("trained-models")
    progress_events = []

    def progress_callback(completed, total, model_name, status):
        progress_events.append((completed, total, model_name, status))

    result = train_and_compare_models(
        prepared_dataset,
        str(model_folder),
        run_id,
        event_logger=event_logger,
        progress_callback=progress_callback,
    )
    db.save_training_results(run_id, result["model_results"], result["best_model"])
    db.log_system_event(
        1,
        "Dataset",
        "upload_completed",
        "Success",
        "127.0.0.1",
        message="Fixture upload completed.",
        run_id=run_id,
        dataset_filename="binary_traffic.csv",
    )
    return {
        "run_id": run_id,
        "result": result,
        "frame": binary_dataframe,
        "prepared": prepared_dataset,
        "progress_events": progress_events,
    }
