import os
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

from services.database_service import (
    DATABASE_FOLDER,
    create_admin,
    create_training_run,
    get_active_deployment,
    get_admin_by_username,
    get_latest_prediction,
    get_latest_training_run,
    get_log_filter_options,
    get_model_result,
    get_training_run,
    initialize_database,
    insert_alert,
    insert_network_traffic_from_flow,
    insert_prediction,
    insert_report,
    list_admins,
    list_alerts,
    list_model_results,
    list_system_logs,
    list_training_runs,
    log_system_event,
    mark_prediction_alert_created,
    save_training_results,
    update_training_run,
    utc_now,
)
from services.deployment_service import DeploymentError, deploy_model
from services.model_registry import MODEL_IDS
from services.preprocessing_service import prepare_dataset
from services.simulation_service import (
    SimulationServiceError,
    get_simulation_schema,
    run_simulation,
)
from services.training_service import train_and_compare_models


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SAVED_MODEL_FOLDER = os.path.join(BASE_DIR, "saved_models")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
ALLOWED_EXTENSIONS = {"csv"}
ADMIN_ROLES = ("Administrator", "Analyst")


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("ALGOGUARD_SECRET_KEY", "algoguard-dev-secret")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("ALGOGUARD_SECURE_COOKIES", "0") == "1"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SAVED_MODEL_FOLDER"] = SAVED_MODEL_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024


def ensure_runtime_folders():
    for folder in (UPLOAD_FOLDER, SAVED_MODEL_FOLDER, REPORT_FOLDER, DATABASE_FOLDER):
        os.makedirs(folder, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(uploaded_file):
    original_name = secure_filename(uploaded_file.filename)
    if not original_name:
        raise ValueError("The uploaded CSV filename is invalid.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    stored_name = f"{timestamp}_{uuid4().hex[:8]}_{original_name}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    uploaded_file.save(file_path)
    return file_path, original_name, stored_name


def save_report(run_id, model_results):
    rows = []
    for result in model_results:
        normalized = result.get("normalized_metrics") or {}
        rows.append(
            {
                "rank": result.get("rank"),
                "model_name": result.get("model_name"),
                "model_type": result.get("model_type"),
                "status": result.get("status"),
                "accuracy": result.get("accuracy"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "f1_score": result.get("f1_score"),
                "roc_auc": result.get("roc_auc"),
                "false_positive_rate": result.get("false_positive_rate"),
                "cpu_time_seconds": result.get("cpu_usage"),
                "peak_ram_increase_mb": result.get("ram_usage"),
                "model_size_mb": result.get("model_size"),
                **{f"normalized_{key}": value for key, value in normalized.items()},
                "overall_score": result.get("overall_score"),
                "error_message": result.get("error_message"),
            }
        )
    report_name = f"training_run_{run_id}_model_results.csv"
    pd.DataFrame(rows).to_csv(
        os.path.join(app.config["REPORT_FOLDER"], report_name),
        index=False,
    )
    return report_name


def current_admin_id():
    return session.get("admin_id")


def current_admin_role():
    return session.get("admin_role")


def _safe_next_url(next_url):
    if not next_url:
        return url_for("dashboard")
    parsed = urlsplit(next_url)
    if parsed.netloc or not next_url.startswith("/"):
        return url_for("dashboard")
    return next_url


def _log(
    module,
    action,
    status,
    message=None,
    run_id=None,
    dataset_filename=None,
    model_name=None,
    prediction_id=None,
):
    return log_system_event(
        current_admin_id(),
        module,
        action,
        status,
        request.remote_addr,
        message=message,
        run_id=run_id,
        dataset_filename=dataset_filename,
        model_name=model_name,
        prediction_id=prediction_id,
    )


def _training_logger(run_id, filename):
    def logger(module, action, status, model_name=None, message=None):
        _log(
            module,
            action,
            status,
            message=message,
            run_id=run_id,
            dataset_filename=filename,
            model_name=model_name,
        )

    return logger


def _execute_prediction(payload, log_module):
    schema = get_simulation_schema()
    deployment = schema.get("deployment") or {}
    _log(
        log_module,
        "prediction_started",
        "Started",
        message="Manual prediction started.",
        run_id=deployment.get("run_id"),
        model_name=deployment.get("model_name"),
    )

    try:
        result = run_simulation(payload)
        traffic_id = insert_network_traffic_from_flow(result["flow_data"], "simulation")
        prediction_id = insert_prediction(
            traffic_id,
            result["model_id"],
            result["prediction"],
            result["confidence"],
            deployment_id=result["deployment_id"],
            model_name=result["deployed_model_name"],
            latency_ms=result["latency_ms"],
            input_payload=result["flow_data"],
        )

        alert_id = None
        if result["prediction"] == "Attack":
            alert_id = insert_alert(
                prediction_id,
                "High",
                f"Attack predicted by {result['deployed_model_name']}.",
            )
            mark_prediction_alert_created(prediction_id)
            _log(
                log_module,
                "attack_predicted",
                "Warning",
                message="Manual traffic record classified as Attack.",
                run_id=result["source_run_id"],
                model_name=result["deployed_model_name"],
                prediction_id=prediction_id,
            )
            _log(
                "Alerts",
                "alert_created",
                "Success",
                message=f"Alert {alert_id} created for prediction {prediction_id}.",
                run_id=result["source_run_id"],
                model_name=result["deployed_model_name"],
                prediction_id=prediction_id,
            )
        else:
            _log(
                log_module,
                "normal_predicted",
                "Success",
                message="Manual traffic record classified as Normal.",
                run_id=result["source_run_id"],
                model_name=result["deployed_model_name"],
                prediction_id=prediction_id,
            )

        result["prediction_id"] = prediction_id
        result["alert_id"] = alert_id
        result["alert_created"] = alert_id is not None
        _log(
            log_module,
            "prediction_completed",
            "Success",
            message=(f"Prediction {prediction_id} completed in {result['latency_ms']:.3f} ms."),
            run_id=result["source_run_id"],
            model_name=result["deployed_model_name"],
            prediction_id=prediction_id,
        )
        return result
    except Exception as error:
        _log(
            log_module,
            "prediction_failed",
            "Failed",
            message=str(error),
            run_id=deployment.get("run_id"),
            model_name=deployment.get("model_name"),
        )
        raise


@app.before_request
def require_admin_login():
    public_endpoints = {"login", "static"}
    if request.endpoint in public_endpoints or request.endpoint is None:
        return None
    if current_admin_id():
        return None
    if request.endpoint == "predict_api":
        return jsonify({"status": "error", "message": "Login required."}), 401
    return redirect(
        url_for("login", next=request.full_path if request.query_string else request.path)
    )


@app.after_request
def prevent_dynamic_page_caching(response):
    """Keep authenticated views out of browser history and intermediary caches."""
    if request.endpoint != "static":
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0, private"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    if current_admin_id():
        _log(
            "Dataset",
            "validation_failed",
            "Failed",
            message="Upload rejected because it exceeds the 256 MB limit.",
        )
    flash("The selected CSV exceeds the 256 MB upload limit.", "danger")
    return redirect(url_for("upload_dataset"))


@app.errorhandler(sqlite3.DatabaseError)
def database_error(error):
    app.logger.error("AlgoGuard database error: %s", error)
    try:
        log_system_event(
            current_admin_id(),
            "Database",
            "database_error",
            "Failed",
            request.remote_addr,
            message="A database operation failed.",
        )
    except sqlite3.DatabaseError:
        pass
    if request.endpoint == "predict_api":
        return jsonify({"status": "error", "message": "A database operation failed."}), 500
    flash("A database operation failed. No destructive recovery was attempted.", "danger")
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_admin_id():
        return redirect(url_for("dashboard"))

    next_url = request.args.get("next") or request.form.get("next") or ""
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        admin = get_admin_by_username(username)
        if admin and check_password_hash(admin["password_hash"], password):
            session.clear()
            session["admin_id"] = admin["admin_id"]
            session["admin_username"] = admin["username"]
            session["admin_role"] = admin["role"]
            _log(
                "Authentication",
                "login_successful",
                "Success",
                message=f"Admin {admin['username']} logged in.",
            )
            flash("Welcome back to AlgoGuard.", "success")
            return redirect(_safe_next_url(next_url))

        log_system_event(
            admin["admin_id"] if admin else None,
            "Authentication",
            "login_failed",
            "Failed",
            request.remote_addr,
            message=f"Failed login attempt for {username or 'unknown'}.",
        )
        flash("Invalid username or password.", "danger")
    return render_template("login.html", next_url=next_url, username=username)


@app.route("/logout")
def logout():
    if current_admin_id():
        _log(
            "Authentication",
            "logout",
            "Success",
            message=f"Admin {session.get('admin_username')} logged out.",
        )
    session.clear()
    return redirect(url_for("login", logged_out="1"))


@app.route("/admins", methods=["GET", "POST"])
def manage_admins():
    if current_admin_role() != "Administrator":
        flash("Only Administrator accounts can manage admin users.", "danger")
        return redirect(url_for("dashboard"))

    form_data = {"username": "", "email": "", "role": "Analyst"}
    if request.method == "POST":
        form_data.update(
            username=request.form.get("username", "").strip(),
            email=request.form.get("email", "").strip(),
            role=request.form.get("role", "Analyst").strip(),
        )
        try:
            password = request.form.get("password", "")
            if password != request.form.get("confirm_password", ""):
                raise ValueError("Passwords do not match.")
            create_admin(form_data["username"], password, form_data["email"], form_data["role"])
            _log(
                "Admin Management",
                "admin_created",
                "Success",
                message=f"Created admin account {form_data['username']}.",
            )
            flash("New admin account created successfully.", "success")
            return redirect(url_for("manage_admins"))
        except ValueError as error:
            _log("Admin Management", "admin_creation_failed", "Failed", message=str(error))
            flash(str(error), "danger")

    return render_template(
        "admins.html",
        admins=list_admins(),
        form_data=form_data,
        role_options=ADMIN_ROLES,
    )


@app.route("/")
def dashboard():
    logs = list_system_logs(page=1, per_page=5)
    latest_run = get_latest_training_run()
    return render_template(
        "dashboard.html",
        latest_run=latest_run,
        model_results=list_model_results(latest_run["run_id"]) if latest_run else [],
        active_deployment=get_active_deployment(),
        latest_prediction=get_latest_prediction(),
        latest_alerts=list_alerts(limit=5),
        latest_logs=logs["items"],
    )


@app.route("/upload", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "GET":
        models = [
            {
                "name": name,
                "type": "Individual" if index < 5 else "Ensemble",
            }
            for index, name in enumerate(MODEL_IDS.values())
        ]
        return render_template("upload.html", models=models)

    uploaded_file = request.files.get("dataset")
    filename = secure_filename(uploaded_file.filename) if uploaded_file else ""
    _log(
        "Dataset",
        "csv_upload_started",
        "Started",
        message=f"CSV upload started for {filename or 'unnamed file'}.",
        dataset_filename=filename or None,
    )

    if not uploaded_file or not uploaded_file.filename:
        _log("Dataset", "validation_failed", "Failed", message="No CSV file selected.")
        flash("Please choose one CSV file before starting analysis.", "danger")
        return redirect(url_for("upload_dataset"))
    if not allowed_file(uploaded_file.filename):
        _log(
            "Dataset",
            "validation_failed",
            "Failed",
            message="Upload rejected because the file is not CSV.",
            dataset_filename=filename,
        )
        flash("Invalid file type. AlgoGuard accepts one CSV file per run.", "danger")
        return redirect(url_for("upload_dataset"))

    run_id = None
    try:
        file_path, original_name, stored_name = save_uploaded_file(uploaded_file)
        run_id = create_training_run(current_admin_id(), original_name, stored_name)
        logger = _training_logger(run_id, original_name)
        _log(
            "Dataset",
            "upload_completed",
            "Success",
            message=f"Uploaded {original_name} as independent run {run_id}.",
            run_id=run_id,
            dataset_filename=original_name,
        )
        _log(
            "Training",
            "training_run_created",
            "Success",
            message=f"Training run {run_id} created.",
            run_id=run_id,
            dataset_filename=original_name,
        )

        update_training_run(run_id, validation_status="in_progress")
        _log(
            "Dataset",
            "validation_started",
            "Started",
            message="CSV validation started.",
            run_id=run_id,
            dataset_filename=original_name,
        )
        update_training_run(run_id, preprocessing_status="in_progress")
        _log(
            "Dataset",
            "preprocessing_started",
            "Started",
            message="Leakage-safe preprocessing preparation started.",
            run_id=run_id,
            dataset_filename=original_name,
        )

        try:
            prepared = prepare_dataset(file_path)
        except ValueError as error:
            update_training_run(
                run_id,
                validation_status="failed",
                preprocessing_status="failed",
                training_status="failed",
                completed_at=utc_now(),
                error_message=str(error),
            )
            _log(
                "Dataset",
                "validation_failed",
                "Failed",
                message=str(error),
                run_id=run_id,
                dataset_filename=original_name,
            )
            _log(
                "Dataset",
                "preprocessing_failed",
                "Failed",
                message="Preprocessing was not completed because validation failed.",
                run_id=run_id,
                dataset_filename=original_name,
            )
            flash(str(error), "danger")
            return redirect(url_for("training_results", run_id=run_id))

        update_training_run(
            run_id,
            row_count=prepared.summary["rows"],
            feature_count=prepared.summary["features"],
            normal_count=prepared.summary["normal_count"],
            anomaly_count=prepared.summary["anomaly_count"],
            target_column=prepared.summary["target_column"],
            class_distribution=prepared.summary["label_counts"],
            validation_status="passed",
            preprocessing_status="completed",
            training_status="in_progress",
        )
        _log(
            "Dataset",
            "validation_passed",
            "Success",
            message="CSV validation passed for binary Normal/Attack training.",
            run_id=run_id,
            dataset_filename=original_name,
        )
        _log(
            "Dataset",
            "preprocessing_completed",
            "Success",
            message="Train/test split and preprocessing schema completed.",
            run_id=run_id,
            dataset_filename=original_name,
        )

        results = train_and_compare_models(
            prepared,
            app.config["SAVED_MODEL_FOLDER"],
            run_id,
            event_logger=logger,
        )
        _log(
            "Training",
            "metric_calculation_completed",
            "Success",
            message="Raw metrics calculated for all completed candidates.",
            run_id=run_id,
            dataset_filename=original_name,
        )
        save_training_results(run_id, results["model_results"], results["best_model"])
        report_name = save_report(run_id, results["model_results"])
        insert_report(current_admin_id(), "Model Performance", run_id=run_id)
        _log(
            "Reports",
            "report_generated",
            "Success",
            message=f"Generated {report_name}.",
            run_id=run_id,
            dataset_filename=original_name,
        )

        if results["best_model"]:
            flash(
                f"Run {run_id} completed. {results['best_model']['model_name']} is recommended.",
                "success",
            )
        else:
            flash("Training finished, but no candidate completed all required metrics.", "warning")
        return redirect(url_for("training_results", run_id=run_id))

    except Exception as error:
        if run_id:
            update_training_run(
                run_id,
                training_status="failed",
                completed_at=utc_now(),
                error_message=str(error),
            )
        _log(
            "Application",
            "application_error",
            "Failed",
            message=f"Training workflow failed: {error}",
            run_id=run_id,
            dataset_filename=filename or None,
        )
        flash(f"AlgoGuard could not complete this run: {error}", "danger")
        return redirect(
            url_for("training_results", run_id=run_id) if run_id else url_for("upload_dataset")
        )


@app.route("/results")
def results():
    latest = get_latest_training_run()
    if not latest:
        flash("Upload one CSV to create a training run.", "warning")
        return redirect(url_for("upload_dataset"))
    return redirect(url_for("training_results", run_id=latest["run_id"]))


@app.route("/runs/<int:run_id>")
def training_results(run_id):
    training_run = get_training_run(run_id)
    if not training_run:
        flash("Training run not found.", "danger")
        return redirect(url_for("training_history"))
    return render_template(
        "results.html",
        training_run=training_run,
        model_results=list_model_results(run_id),
        active_deployment=get_active_deployment(),
    )


@app.route("/history")
def training_history():
    page = request.args.get("page", 1, type=int)
    return render_template("history.html", runs=list_training_runs(page=page, per_page=20))


@app.route("/runs/<int:run_id>/deploy/<int:model_id>", methods=["POST"])
def deploy_training_model(run_id, model_id):
    model = get_model_result(model_id)
    if not model or model.get("run_id") != run_id:
        flash("The selected model does not belong to this training run.", "danger")
        return redirect(url_for("training_results", run_id=run_id))

    _log(
        "Deployment",
        "deployment_started",
        "Started",
        message=f"Deployment started for {model['model_name']}.",
        run_id=run_id,
        dataset_filename=model.get("filename"),
        model_name=model["model_name"],
    )
    try:
        deployed = deploy_model(model_id, current_admin_id())
        previous = deployed["deployment"].get("previous")
        if previous:
            _log(
                "Deployment",
                "previous_deployed_model_replaced",
                "Success",
                message=f"Previous deployment {previous['deployment_id']} was replaced.",
                run_id=run_id,
                model_name=model["model_name"],
            )
        _log(
            "Deployment",
            "model_deployed",
            "Success",
            message=f"{model['model_name']} is now the active model.",
            run_id=run_id,
            dataset_filename=model.get("filename"),
            model_name=model["model_name"],
        )
        flash(f"{model['model_name']} is now deployed.", "success")
    except DeploymentError as error:
        _log(
            "Deployment",
            "deployment_failed",
            "Failed",
            message=str(error),
            run_id=run_id,
            model_name=model["model_name"],
        )
        flash(str(error), "danger")
    return redirect(url_for("training_results", run_id=run_id))


@app.route("/simulation", methods=["GET", "POST"])
def simulation_demo():
    schema = get_simulation_schema()
    form_data = {field["name"]: field.get("default", "") for field in schema.get("fields", [])}
    result = None
    if request.method == "POST":
        form_data.update(request.form.to_dict())
        try:
            result = _execute_prediction(form_data, "Prediction")
        except SimulationServiceError as error:
            flash(str(error), "danger")
        except Exception as error:
            flash(f"Prediction failed safely: {error}", "danger")
    return render_template(
        "simulation.html",
        schema=schema,
        form_data=form_data,
        result=result,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        result = _execute_prediction(request.get_json(silent=True) or {}, "Prediction API")
        return jsonify({"status": "success", **result})
    except SimulationServiceError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    except Exception as error:
        return jsonify({"status": "error", "message": str(error)}), 500


@app.route("/alerts")
def alert_history():
    return render_template("alerts.html", alerts=list_alerts(limit=250))


@app.route("/logs")
def system_logs():
    filters = {
        "date_from": request.args.get("date_from", "").strip(),
        "date_to": request.args.get("date_to", "").strip(),
        "module": request.args.get("module", "").strip(),
        "status": request.args.get("status", "").strip(),
        "model": request.args.get("model", "").strip(),
        "run_id": request.args.get("run_id", "").strip(),
        "search": request.args.get("search", "").strip(),
    }
    page = request.args.get("page", 1, type=int)
    return render_template(
        "logs.html",
        logs=list_system_logs(filters, page=page, per_page=25),
        filters=filters,
        filter_options=get_log_filter_options(),
    )


ensure_runtime_folders()
initialize_database()


if __name__ == "__main__":
    port = int(os.environ.get("ALGOGUARD_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
