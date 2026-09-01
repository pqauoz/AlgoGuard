import os
import secrets
import sqlite3
from urllib.parse import urlsplit

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash

from services.database_service import (
    DATABASE_FOLDER,
    create_admin,
    get_active_deployment,
    get_admin_by_username,
    get_detection_stats,
    get_latest_prediction,
    get_log_filter_options,
    initialize_database,
    insert_alert,
    insert_network_traffic_from_flow,
    insert_prediction,
    list_admins,
    list_alerts,
    list_system_logs,
    log_system_event,
    mark_prediction_alert_created,
)
from services.live_monitor_service import (
    LiveMonitorError,
    get_options,
    get_status,
    pause_session,
    resume_session,
    start_session,
    stop_session,
)
from services.simulation_service import (
    SimulationServiceError,
    get_simulation_schema,
    run_simulation,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAVED_MODEL_FOLDER = os.path.join(BASE_DIR, "saved_models")
ADMIN_ROLES = ("Administrator", "Analyst")


app = Flask(__name__)
_configured_secret = os.environ.get("ALGOGUARD_SECRET_KEY")
app.config["SECRET_KEY"] = _configured_secret or secrets.token_hex(32)
app.config["SECRET_KEY_EPHEMERAL"] = _configured_secret is None
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("ALGOGUARD_SECURE_COOKIES", "0") == "1"
app.config["SAVED_MODEL_FOLDER"] = SAVED_MODEL_FOLDER


def ensure_runtime_folders():
    capture_folder = os.path.join(BASE_DIR, "captures")
    for folder in (SAVED_MODEL_FOLDER, DATABASE_FOLDER, capture_folder):
        os.makedirs(folder, exist_ok=True)


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


JSON_ENDPOINTS = {
    "predict_api",
    "monitor_start",
    "monitor_pause",
    "monitor_resume",
    "monitor_stop",
    "monitor_status",
}


@app.before_request
def require_admin_login():
    public_endpoints = {"login", "static"}
    if request.endpoint in public_endpoints or request.endpoint is None:
        return None
    if current_admin_id():
        return None
    if request.endpoint in JSON_ENDPOINTS:
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
    if request.endpoint in JSON_ENDPOINTS:
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
    return render_template(
        "dashboard.html",
        active_deployment=get_active_deployment(),
        stats=get_detection_stats(),
        latest_prediction=get_latest_prediction(),
        latest_alerts=list_alerts(limit=5),
        latest_logs=logs["items"],
    )


@app.route("/monitor")
def live_monitor():
    schema = get_simulation_schema()
    return render_template(
        "monitor.html",
        available=schema.get("available", False),
        unavailable_message=schema.get("message"),
        deployment=schema.get("deployment"),
        options=get_options(),
        initial_status=get_status(),
    )


def _monitor_json(callback, action, success_message):
    """Run one monitor control action and answer the page with JSON."""
    try:
        monitor_session = callback()
    except LiveMonitorError as error:
        _log("Live Monitor", f"{action}_failed", "Failed", message=str(error))
        return jsonify({"status": "error", "message": str(error)}), 409
    _log("Live Monitor", action, "Success", message=success_message)
    return jsonify({"status": "success", "session": monitor_session})


@app.route("/monitor/start", methods=["POST"])
def monitor_start():
    payload = request.get_json(silent=True) or {}
    defaults = get_options()["defaults"]
    return _monitor_json(
        lambda: start_session(
            current_admin_id(),
            source_type=payload.get("source_type", defaults["source_type"]),
            dataset=payload.get("dataset", defaults["dataset"]),
            capture_file=payload.get("capture_file"),
            interface=payload.get("interface"),
            speed=payload.get("speed", defaults["speed"]),
            order=payload.get("order", defaults["order"]),
            persist=payload.get("persist", defaults["persist"]),
        ),
        "monitor_start_requested",
        "Live monitoring session requested.",
    )


@app.route("/monitor/pause", methods=["POST"])
def monitor_pause():
    return _monitor_json(pause_session, "monitor_paused", "Live monitoring paused.")


@app.route("/monitor/resume", methods=["POST"])
def monitor_resume():
    return _monitor_json(resume_session, "monitor_resumed", "Live monitoring resumed.")


@app.route("/monitor/stop", methods=["POST"])
def monitor_stop():
    return _monitor_json(stop_session, "monitor_stop_requested", "Live monitoring stopped.")


@app.route("/monitor/status")
def monitor_status():
    return jsonify({"status": "success", **get_status(request.args.get("since", 0))})


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
    port = int(os.environ.get("ALGOGUARD_PORT", "5000"))
    # Safe local defaults: the debugger stays off and the server listens only on
    # this machine, because AlgoGuard now handles captured network traffic.
    # Set FLASK_DEBUG=1 while developing, and ALGOGUARD_HOST=0.0.0.0 only on a
    # network you control and intend to expose this prototype to.
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("ALGOGUARD_HOST", "127.0.0.1")
    # threaded=True keeps the live monitor's status polling responsive while a
    # monitoring session is classifying flows in its background thread.
    app.run(debug=debug, host=host, port=port, use_reloader=False, threaded=True)
