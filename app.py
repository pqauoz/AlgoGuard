import os
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

from services.database_service import (
    DATABASE_FOLDER,
    create_admin,
    get_admin_by_username,
    get_or_create_simulation_model,
    initialize_database,
    insert_alert,
    insert_detection_model,
    insert_network_traffic_from_flow,
    insert_prediction,
    insert_report,
    list_admins,
    log_system_event,
)
from services.preprocessing_service import prepare_dataset
from services.simulation_service import SimulationServiceError, run_simulation
from services.training_service import train_and_compare_models


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SAVED_MODEL_FOLDER = os.path.join(BASE_DIR, "saved_models")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
ALLOWED_EXTENSIONS = {"csv"}
ADMIN_ROLES = ("Administrator", "Analyst")


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("ALGOGUARD_SECRET_KEY", "algoguard-dev-secret")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SAVED_MODEL_FOLDER"] = SAVED_MODEL_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER


def ensure_runtime_folders():
    """Create folders used for uploads, model artifacts, reports, and SQLite."""
    for folder in (UPLOAD_FOLDER, SAVED_MODEL_FOLDER, REPORT_FOLDER, DATABASE_FOLDER):
        os.makedirs(folder, exist_ok=True)


def allowed_file(filename):
    """Return True when the uploaded file has a CSV extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(uploaded_file):
    """Save a CSV upload with a unique name to avoid overwriting older runs."""
    original_name = secure_filename(uploaded_file.filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    unique_name = f"{timestamp}_{uuid4().hex[:8]}_{original_name}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    uploaded_file.save(file_path)
    return file_path, original_name


def save_report(results):
    """Write a small CSV report for the latest model comparison."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    report_name = f"model_report_{timestamp}.csv"
    report_path = os.path.join(app.config["REPORT_FOLDER"], report_name)
    pd.DataFrame(results["model_results"]).to_csv(report_path, index=False)
    return report_name


def default_simulation_form():
    """Return starter values for the manual network flow simulation."""
    return {
        "dur": "0.001",
        "rate": "15",
        "spkts": "12",
        "dpkts": "8",
        "sbytes": "1500",
        "dbytes": "800",
        "sttl": "64",
        "dttl": "64",
        "proto": "tcp",
        "state": "CON",
    }


def current_admin_id():
    """Return the logged-in admin id, if available."""
    return session.get("admin_id")


def current_admin_role():
    """Return the logged-in admin role."""
    return session.get("admin_role")


def persist_training_records(results):
    """Save model comparison and report metadata to SQLite."""
    for model_result in results["model_results"]:
        insert_detection_model(model_result)

    insert_report(current_admin_id(), "Model Performance")


def persist_simulation_record(result):
    """Save a manual simulation prediction and create an alert for attacks."""
    traffic_id = insert_network_traffic_from_flow(result["flow_data"], "simulation")
    model_id = get_or_create_simulation_model()
    prediction_id = insert_prediction(
        traffic_id,
        model_id,
        result["prediction"],
        result.get("confidence"),
    )

    if result["prediction"] == "Attack":
        insert_alert(
            prediction_id,
            "High",
            "Anomalous traffic detected by the pretrained simulation model.",
        )

    return prediction_id


@app.before_request
def require_admin_login():
    """Protect all pages except login and static assets."""
    public_endpoints = {"login", "static"}

    if request.endpoint in public_endpoints or request.endpoint is None:
        return None

    if current_admin_id():
        return None

    if request.endpoint == "predict_api":
        return jsonify({"status": "error", "message": "Login required."}), 401

    return redirect(url_for("login", next=request.full_path if request.query_string else request.path))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Authenticate an admin account before allowing dashboard access."""
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

            log_system_event(
                admin["admin_id"],
                "Authentication",
                "Admin login",
                "Success",
                request.remote_addr,
            )

            flash("Welcome back to AlgoGuard.", "success")
            if not next_url.startswith("/"):
                next_url = url_for("dashboard")
            return redirect(next_url or url_for("dashboard"))

        log_system_event(
            admin["admin_id"] if admin else None,
            "Authentication",
            f"Failed login for {username or 'unknown'}",
            "Failed",
            request.remote_addr,
        )
        flash("Invalid username or password.", "danger")

    return render_template("login.html", next_url=next_url, username=username)


@app.route("/logout")
def logout():
    """End the current admin session."""
    admin_id = current_admin_id()
    if admin_id:
        log_system_event(
            admin_id,
            "Authentication",
            "Admin logout",
            "Success",
            request.remote_addr,
        )

    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/admins", methods=["GET", "POST"])
def manage_admins():
    """Allow administrators to create additional admin accounts."""
    if current_admin_role() != "Administrator":
        flash("Only Administrator accounts can manage admin users.", "danger")
        return redirect(url_for("dashboard"))

    form_data = {
        "username": "",
        "email": "",
        "role": "Analyst",
    }

    if request.method == "POST":
        form_data.update(
            {
                "username": request.form.get("username", "").strip(),
                "email": request.form.get("email", "").strip(),
                "role": request.form.get("role", "Analyst").strip(),
            }
        )
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        try:
            if password != confirm_password:
                raise ValueError("Passwords do not match.")

            create_admin(
                form_data["username"],
                password,
                form_data["email"],
                form_data["role"],
            )
            log_system_event(
                current_admin_id(),
                "Admin Management",
                f"Created admin account: {form_data['username']}",
                "Success",
                request.remote_addr,
            )
            flash("New admin account created successfully.", "success")
            return redirect(url_for("manage_admins"))

        except ValueError as error:
            log_system_event(
                current_admin_id(),
                "Admin Management",
                f"Create admin failed: {error}",
                "Failed",
                request.remote_addr,
            )
            flash(str(error), "danger")

    return render_template(
        "admins.html",
        admins=list_admins(),
        form_data=form_data,
        role_options=ADMIN_ROLES,
    )


@app.route("/")
def dashboard():
    """Show the main AlgoGuard dashboard."""
    return render_template("dashboard.html", latest_run=session.get("latest_run"))


@app.route("/upload", methods=["GET", "POST"])
def upload_dataset():
    """Handle CSV uploads and run the full training workflow."""
    if request.method == "GET":
        return render_template("upload.html")

    uploaded_file = request.files.get("dataset")

    if not uploaded_file or uploaded_file.filename == "":
        flash("Please choose a CSV file before starting analysis.", "danger")
        return redirect(url_for("upload_dataset"))

    if not allowed_file(uploaded_file.filename):
        flash("Invalid file type. AlgoGuard accepts CSV files only.", "danger")
        return redirect(url_for("upload_dataset"))

    try:
        file_path, original_name = save_uploaded_file(uploaded_file)
        prepared_dataset = prepare_dataset(file_path)
        results = train_and_compare_models(prepared_dataset, app.config["SAVED_MODEL_FOLDER"])
        report_name = save_report(results)
        persist_training_records(results)

        session["latest_run"] = {
            "filename": original_name,
            "uploaded_at": datetime.now().strftime("%b %d, %Y %I:%M %p"),
            "report_name": report_name,
            "dataset": prepared_dataset.summary,
            "model_results": results["model_results"],
            "best_model": results["best_model"],
        }

        log_system_event(
            current_admin_id(),
            "Dataset",
            f"Uploaded and trained dataset: {original_name}",
            "Success",
            request.remote_addr,
        )

        flash("Dataset processed successfully. Models were trained and compared.", "success")
        return redirect(url_for("results"))

    except ValueError as error:
        log_system_event(
            current_admin_id(),
            "Dataset",
            f"Dataset validation failed: {error}",
            "Failed",
            request.remote_addr,
        )
        flash(str(error), "danger")
        return redirect(url_for("upload_dataset"))
    except Exception as error:
        log_system_event(
            current_admin_id(),
            "Dataset",
            f"Dataset processing failed: {error}",
            "Failed",
            request.remote_addr,
        )
        flash(f"AlgoGuard could not process this dataset: {error}", "danger")
        return redirect(url_for("upload_dataset"))


@app.route("/results")
def results():
    """Display the latest training and evaluation results."""
    latest_run = session.get("latest_run")
    if not latest_run:
        flash("Upload a dataset first to generate anomaly detection results.", "warning")
        return redirect(url_for("upload_dataset"))

    return render_template("results.html", latest_run=latest_run)


@app.route("/simulation", methods=["GET", "POST"])
def simulation_demo():
    """Run a manual traffic-flow simulation using the pretrained winning model."""
    form_data = default_simulation_form()
    result = None

    if request.method == "POST":
        form_data.update(request.form.to_dict())
        try:
            result = run_simulation(form_data)
            persist_simulation_record(result)
            log_system_event(
                current_admin_id(),
                "Simulation",
                f"Manual flow predicted as {result['prediction']}",
                "Success",
                request.remote_addr,
            )
        except SimulationServiceError as error:
            log_system_event(
                current_admin_id(),
                "Simulation",
                f"Simulation model failed: {error}",
                "Failed",
                request.remote_addr,
            )
            flash(str(error), "danger")
        except Exception as error:
            log_system_event(
                current_admin_id(),
                "Simulation",
                f"Simulation failed: {error}",
                "Failed",
                request.remote_addr,
            )
            flash(f"Simulation failed: {error}", "danger")

    return render_template("simulation.html", form_data=form_data, result=result)


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON endpoint for manual simulation predictions."""
    try:
        payload = request.get_json(silent=True) or {}
        result = run_simulation(payload)
        persist_simulation_record(result)
        log_system_event(
            current_admin_id(),
            "Simulation API",
            f"API flow predicted as {result['prediction']}",
            "Success",
            request.remote_addr,
        )
        return jsonify({"status": "success", **result})
    except Exception as error:
        log_system_event(
            current_admin_id(),
            "Simulation API",
            f"API prediction failed: {error}",
            "Failed",
            request.remote_addr,
        )
        return jsonify({"status": "error", "message": str(error)}), 500


ensure_runtime_folders()
initialize_database()


if __name__ == "__main__":
    port = int(os.environ.get("ALGOGUARD_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
