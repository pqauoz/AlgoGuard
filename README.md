# AlgoGuard

AlgoGuard is a Flask-based research prototype for lightweight network anomaly detection. It is designed as a capstone-friendly simulation system for analyzing CSV network traffic datasets, training ensemble machine learning models, comparing their performance, and running a manual traffic-flow demo.

The current version is a simulation/prototype. It analyzes uploaded datasets or manually entered traffic-flow values. It does not capture live packets, monitor a real network continuously, or block threats automatically.

---

## Features

- Dark cybersecurity dashboard
- Admin login page with password hashing
- Multiple admin accounts through an Admins page
- CSV dataset upload
- Automatic preprocessing
- Categorical feature encoding
- Numerical feature scaling
- Train/test split
- Ensemble model training and comparison
- Manual single-flow simulation demo using pretrained model artifacts
- Performance metrics table
- Normal traffic and anomaly counts
- Best model identification
- Joblib model saving
- SQLite storage for admins, model records, predictions, alerts, reports, and system logs
- Basic report export

---

## Models

The main dataset workflow trains and compares:

- Random Forest
- Gradient Boosting
- Voting Classifier

The simulation demo uses the pretrained model artifacts in `legacy_simulation/`.

---

## Metrics

AlgoGuard displays:

- Accuracy
- Precision
- Recall
- F1-score
- False Positive Rate
- Processing Time
- Predicted normal count
- Predicted anomaly count

---

## Project Structure

```text
AlgoGuard/
|-- app.py
|-- requirements.txt
|-- services/
|   |-- database_service.py
|   |-- preprocessing_service.py
|   |-- training_service.py
|   |-- evaluation_service.py
|   `-- simulation_service.py
|-- templates/
|   |-- admins.html
|   |-- base.html
|   |-- dashboard.html
|   |-- login.html
|   |-- upload.html
|   |-- results.html
|   `-- simulation.html
|-- static/
|   |-- css/
|   |   `-- style.css
|   `-- js/
|-- uploads/
|-- saved_models/
|-- reports/
|-- database/
|-- legacy_simulation/
|   |-- app.py
|   |-- feature_extractor.py
|   |-- flow_builder.py
|   |-- model_loader.py
|   |-- predictor.py
|   |-- preprocessor.py
|   `-- templates/
|-- dataset/
|-- models/
|-- notebooks/
`-- results/
```

---

## Folder Guide

- `app.py`: main Flask application entrypoint.
- `services/`: database, preprocessing, training, evaluation, and simulation logic.
- `templates/`: pages used by the main Flask app.
- `static/`: CSS and JavaScript assets.
- `uploads/`: temporary uploaded CSV files.
- `saved_models/`: Joblib models created by the training workflow.
- `reports/`: CSV reports created after model comparison.
- `database/`: local SQLite database file created when the app runs.
- `legacy_simulation/`: older pretrained manual prediction demo files used by the Simulation page.
- `dataset/`, `models/`, `notebooks/`, `results/`: research and experiment artifacts.

---

## System Requirements

- Python 3.8 or higher
- Git

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pqauoz/AlgoGuard.git
cd AlgoGuard
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the main Flask app from the project root:

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

Default local login:

```text
Username: admin
Password: admin123
```

After logging in with an Administrator account, open the Admins page to create
additional admin accounts.

You can change the seeded local admin before the first run by setting:

```bash
set ALGOGUARD_ADMIN_USERNAME=your_username
set ALGOGUARD_ADMIN_PASSWORD=your_password
set ALGOGUARD_ADMIN_EMAIL=your_email@example.com
python app.py
```

If port 5000 is already in use:

```bash
set ALGOGUARD_PORT=5001
python app.py
```

Then open:

```text
http://localhost:5001
```

---

## Using AlgoGuard

### Login

Open the app and sign in with the seeded admin account. After login, the
dashboard, upload workflow, simulation page, and results page become available.
Login attempts, logout events, dataset training, and simulation predictions are
recorded in the SQLite `system_log` table.

Administrator accounts can create additional admin users from the Admins page.
The new accounts are stored in SQLite with hashed passwords.

### Dataset Mode

1. Open the Upload page.
2. Upload a CSV network traffic dataset.
3. Make sure the final column is the target label.
4. AlgoGuard preprocesses the data automatically.
5. AlgoGuard trains and compares the three ensemble models.
6. Results are displayed in a table and trained models are saved in `saved_models/`.
7. Model metrics and report metadata are saved in the SQLite database.

### Simulation Demo

Open the Simulation page to manually enter one network traffic flow and test it against the pretrained model artifacts in `legacy_simulation/`.
Each simulation stores a traffic record and prediction in SQLite. If the result
is Attack, AlgoGuard also creates an alert record.

Example normal-style input:

| Feature | Value |
| --- | --- |
| Duration | 0.001 |
| Rate | 15 |
| Protocol | tcp |
| State | CON |

Example attack-style input:

| Feature | Value |
| --- | --- |
| Duration | 0.000001 |
| Rate | 5000 |
| Protocol | tcp |
| State | REQ |

---

## Simulation Scope

AlgoGuard is currently a research prototype and simulation. It demonstrates how machine learning can detect anomalous traffic patterns from prepared data.

This version does not:

- Capture live packets from a real network
- Monitor routers, switches, or endpoints continuously
- Block attacks
- Send alerts to security tools
- Replace a production IDS or IPS

Future work could add live packet capture, scheduled monitoring, alerting, and deployment features for real network environments.

---

## Troubleshooting

### ModuleNotFoundError

Activate your virtual environment and reinstall dependencies:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Invalid CSV

The uploaded file must:

- Be a `.csv` file
- Contain at least one feature column
- Use the last column as the target label
- Contain at least two target classes
- Have enough rows to split into training and testing sets

### Model File Not Found

The Simulation page uses legacy pretrained artifacts. Confirm these files exist in `legacy_simulation/`:

- `Stacking_Top3_GB.pkl` or `Stacking_Top3_GB.zip`
- `scaler.pkl`
- `encoded_columns.npy`

### Forgot Local Admin Password

The default password is only seeded when the SQLite database is first created.
For local development, stop the app, delete `database/algoguard.sqlite3`, set new
admin environment variables if needed, and run `python app.py` again.

---

## License

This project is intended for academic and research purposes.
