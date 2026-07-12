import os
import sqlite3
from datetime import datetime, timezone

from werkzeug.security import generate_password_hash


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATABASE_FOLDER = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_FOLDER, "algoguard.sqlite3")
SIMULATION_MODEL_NAME = os.environ.get("ALGOGUARD_SIMULATION_MODEL", "Stacking_Top3_LR")


def utc_now():
    """Return a database-friendly UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_connection():
    """Open a SQLite connection with dictionary-like rows."""
    os.makedirs(DATABASE_FOLDER, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create all AlgoGuard database tables and seed a default admin."""
    os.makedirs(DATABASE_FOLDER, exist_ok=True)

    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS admin (
                admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                email TEXT,
                role TEXT NOT NULL DEFAULT 'Administrator',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS network_traffic (
                traffic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_ip TEXT,
                destination_ip TEXT,
                source_port INTEGER,
                destination_port INTEGER,
                protocol TEXT,
                packet_size INTEGER,
                flags TEXT,
                dataset_source TEXT
            );

            CREATE TABLE IF NOT EXISTS detection_model (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT,
                training_date TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                specificity REAL,
                fpr REAL,
                roc_auc REAL,
                cpu_usage REAL,
                ram_usage REAL,
                training_time REAL,
                model_size REAL
            );

            CREATE TABLE IF NOT EXISTS prediction (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                traffic_id INTEGER NOT NULL,
                model_id INTEGER NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence_score REAL,
                prediction_timestamp TEXT NOT NULL,
                FOREIGN KEY (traffic_id) REFERENCES network_traffic (traffic_id),
                FOREIGN KEY (model_id) REFERENCES detection_model (model_id)
            );

            CREATE TABLE IF NOT EXISTS alert (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                severity_level TEXT NOT NULL,
                alert_status TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                description TEXT,
                FOREIGN KEY (prediction_id) REFERENCES prediction (prediction_id)
            );

            CREATE TABLE IF NOT EXISTS report (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id INTEGER,
                report_type TEXT NOT NULL,
                date_range_start TEXT,
                date_range_end TEXT,
                generated_at TEXT NOT NULL,
                FOREIGN KEY (admin_id) REFERENCES admin (admin_id)
            );

            CREATE TABLE IF NOT EXISTS report_alert (
                report_id INTEGER NOT NULL,
                alert_id INTEGER NOT NULL,
                PRIMARY KEY (report_id, alert_id),
                FOREIGN KEY (report_id) REFERENCES report (report_id),
                FOREIGN KEY (alert_id) REFERENCES alert (alert_id)
            );

            CREATE TABLE IF NOT EXISTS system_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id INTEGER,
                module TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                ip_address TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (admin_id) REFERENCES admin (admin_id)
            );
            """
        )

    seed_default_admin()


def seed_default_admin():
    """Create the local demo admin account when no admin exists yet."""
    username = os.environ.get("ALGOGUARD_ADMIN_USERNAME", "admin")
    password = os.environ.get("ALGOGUARD_ADMIN_PASSWORD", "admin123")
    email = os.environ.get("ALGOGUARD_ADMIN_EMAIL", "admin@algoguard.local")

    with get_connection() as connection:
        admin_count = connection.execute("SELECT COUNT(*) FROM admin").fetchone()[0]
        if admin_count:
            return

        connection.execute(
            """
            INSERT INTO admin (username, password_hash, email, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, generate_password_hash(password), email, "Administrator", utc_now()),
        )


def get_admin_by_username(username):
    """Fetch an admin account by username."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM admin WHERE LOWER(username) = LOWER(?)",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def list_admins():
    """Return all admin accounts for the management page."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT admin_id, username, email, role, created_at
            FROM admin
            ORDER BY admin_id
            """
        ).fetchall()
        return [dict(row) for row in rows]


def create_admin(username, password, email="", role="Administrator"):
    """Create a new admin account with a hashed password."""
    username = str(username or "").strip()
    password = str(password or "")
    email = str(email or "").strip()
    role = str(role or "Administrator").strip()

    if not username:
        raise ValueError("Username is required.")

    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")

    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    if role not in {"Administrator", "Analyst"}:
        raise ValueError("Invalid admin role.")

    with get_connection() as connection:
        existing_admin = connection.execute(
            "SELECT admin_id FROM admin WHERE LOWER(username) = LOWER(?)",
            (username,),
        ).fetchone()

        if existing_admin:
            raise ValueError("An admin with that username already exists.")

        cursor = connection.execute(
            """
            INSERT INTO admin (username, password_hash, email, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, generate_password_hash(password), email, role, utc_now()),
        )
        return cursor.lastrowid


def log_system_event(admin_id, module, action, status, ip_address=None):
    """Write an audit trail entry to system_log."""
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO system_log (admin_id, module, action, status, ip_address, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (admin_id, module, action, status, ip_address, utc_now()),
        )


def insert_detection_model(model_result, version="dataset-workflow"):
    """Persist one model evaluation row from the training workflow."""
    fpr = _safe_float(model_result.get("false_positive_rate"))
    specificity = round(100 - fpr, 2) if fpr is not None else None
    model_path = model_result.get("model_path")
    model_size = None

    if model_path and os.path.exists(model_path):
        model_size = round(os.path.getsize(model_path) / 1024, 2)

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO detection_model (
                model_name, version, training_date, accuracy, precision_score,
                recall, f1_score, specificity, fpr, roc_auc, cpu_usage,
                ram_usage, training_time, model_size
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_result.get("model_name"),
                version,
                utc_now(),
                _safe_float(model_result.get("accuracy")),
                _safe_float(model_result.get("precision")),
                _safe_float(model_result.get("recall")),
                _safe_float(model_result.get("f1_score")),
                specificity,
                fpr,
                None,
                None,
                None,
                _safe_float(model_result.get("processing_time")),
                model_size,
            ),
        )
        return cursor.lastrowid


def get_or_create_simulation_model():
    """Return the database id for the pretrained simulation model."""
    model_name = SIMULATION_MODEL_NAME
    version = "pretrained-simulation"

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT model_id FROM detection_model
            WHERE model_name = ? AND version = ?
            ORDER BY model_id DESC
            LIMIT 1
            """,
            (model_name, version),
        ).fetchone()

        if row:
            return row["model_id"]

        cursor = connection.execute(
            """
            INSERT INTO detection_model (model_name, version, training_date)
            VALUES (?, ?, ?)
            """,
            (model_name, version, utc_now()),
        )
        return cursor.lastrowid


def insert_network_traffic_from_flow(flow_data, dataset_source="simulation"):
    """Store one traffic-flow record using the database design fields."""
    source_bytes = _safe_float(flow_data.get("sbytes"), 0)
    destination_bytes = _safe_float(flow_data.get("dbytes"), 0)
    packet_size = _safe_int(
        flow_data.get("packet_size"),
        int(source_bytes + destination_bytes),
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO network_traffic (
                timestamp, source_ip, destination_ip, source_port,
                destination_port, protocol, packet_size, flags, dataset_source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(flow_data.get("timestamp") or utc_now()),
                str(flow_data.get("source_ip") or "manual"),
                str(flow_data.get("destination_ip") or "manual"),
                _safe_int(flow_data.get("source_port"), 0),
                _safe_int(flow_data.get("destination_port"), 0),
                str(flow_data.get("protocol") or flow_data.get("proto") or ""),
                packet_size,
                str(flow_data.get("flags") or flow_data.get("state") or ""),
                dataset_source,
            ),
        )
        return cursor.lastrowid


def insert_prediction(traffic_id, model_id, predicted_label, confidence_score=None):
    """Store one prediction record."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction (
                traffic_id, model_id, predicted_label, confidence_score,
                prediction_timestamp
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                traffic_id,
                model_id,
                str(predicted_label),
                _safe_float(confidence_score),
                utc_now(),
            ),
        )
        return cursor.lastrowid


def insert_alert(prediction_id, severity_level, description, alert_status="Open"):
    """Store an alert for an anomalous prediction."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO alert (
                prediction_id, severity_level, alert_status, detected_at, description
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (prediction_id, severity_level, alert_status, utc_now(), description),
        )
        return cursor.lastrowid


def insert_report(admin_id, report_type, date_range_start=None, date_range_end=None):
    """Store a generated report record."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO report (
                admin_id, report_type, date_range_start, date_range_end, generated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (admin_id, report_type, date_range_start, date_range_end, utc_now()),
        )
        return cursor.lastrowid


def _safe_float(value, default=None):
    """Convert values from forms, pandas, or NumPy into SQLite-friendly floats."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    """Convert values from forms, pandas, or NumPy into SQLite-friendly integers."""
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default
