import os
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from services.preprocessing_service import prepare_dataset
from services.simulation_service import SimulationServiceError, run_simulation
from services.training_service import train_and_compare_models


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SAVED_MODEL_FOLDER = os.path.join(BASE_DIR, "saved_models")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
ALLOWED_EXTENSIONS = {"csv"}


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("ALGOGUARD_SECRET_KEY", "algoguard-dev-secret")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SAVED_MODEL_FOLDER"] = SAVED_MODEL_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER


def ensure_runtime_folders():
    """Create folders used for uploads, model artifacts, and reports."""
    for folder in (UPLOAD_FOLDER, SAVED_MODEL_FOLDER, REPORT_FOLDER):
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

        session["latest_run"] = {
            "filename": original_name,
            "uploaded_at": datetime.now().strftime("%b %d, %Y %I:%M %p"),
            "report_name": report_name,
            "dataset": prepared_dataset.summary,
            "model_results": results["model_results"],
            "best_model": results["best_model"],
        }

        flash("Dataset processed successfully. Models were trained and compared.", "success")
        return redirect(url_for("results"))

    except ValueError as error:
        flash(str(error), "danger")
        return redirect(url_for("upload_dataset"))
    except Exception as error:
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
    """Run a manual traffic-flow simulation using the pretrained legacy model."""
    form_data = default_simulation_form()
    result = None

    if request.method == "POST":
        form_data.update(request.form.to_dict())
        try:
            result = run_simulation(form_data)
        except SimulationServiceError as error:
            flash(str(error), "danger")
        except Exception as error:
            flash(f"Simulation failed: {error}", "danger")

    return render_template("simulation.html", form_data=form_data, result=result)


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON endpoint for manual simulation predictions."""
    try:
        payload = request.get_json(silent=True) or {}
        result = run_simulation(payload)
        return jsonify({"status": "success", **result})
    except Exception as error:
        return jsonify({"status": "error", "message": str(error)}), 500


if __name__ == "__main__":
    ensure_runtime_folders()
    port = int(os.environ.get("ALGOGUARD_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
