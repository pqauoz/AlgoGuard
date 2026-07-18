# AlgoGuard

AlgoGuard is a local Flask application for binary network traffic classification. One uploaded CSV creates one independent training run, evaluates exactly seven Scikit-learn approaches, recommends a winner using nine normalized metrics, and lets an administrator explicitly deploy that model for manual single-record prediction.

AlgoGuard does not capture live packets, monitor a network continuously, or block traffic. It is an operational research prototype for prepared CSV data and manual prediction.

## Runtime Architecture

Every valid upload trains these exact candidates:

1. Random Forest
2. Gradient Boosting
3. AdaBoost
4. K-Nearest Neighbors
5. Naive Bayes
6. Soft Voting Ensemble using all five individual estimators
7. Stacking Ensemble using all five base estimators and Logistic Regression as the final estimator

Each candidate receives its own unfitted estimator instances. All candidates use the same stratified train/test split. A Scikit-learn pipeline fits missing-value handling, scaling, and categorical encoding only on training data, then saves preprocessing and classification together in the Joblib artifact.

## Ranking

Higher is better for Accuracy, Precision, Recall, F1-score, and ROC-AUC. Lower is better for False Positive Rate, process CPU time, peak process RAM increase, and complete artifact size.

AlgoGuard min-max normalizes all nine metrics across valid candidates. The overall score is their equal arithmetic average. Invalid or incomplete candidates are excluded. Ties are resolved by F1, Recall, ROC-AUC, FPR, RAM, model size, then model name.

## Main Workflow

```text
CSV upload
  -> validation and independent training-run record
  -> stratified split
  -> seven leakage-safe pipelines
  -> raw and normalized metrics
  -> recommendation
  -> administrator deployment
  -> manual prediction using only the active artifact
  -> prediction, optional alert, and system logs
```

Training history, results, deployments, predictions, alerts, reports, admin accounts, and audit events are stored in SQLite. Uploaded files, reports, per-run artifacts, and the active artifact are runtime files excluded from Git.

## Project Structure

```text
AlgoGuard/
|-- app.py
|-- migrate.py
|-- requirements.txt
|-- requirements-dev.txt
|-- services/
|   |-- database_service.py
|   |-- deployment_service.py
|   |-- evaluation_service.py
|   |-- model_registry.py
|   |-- preprocessing_service.py
|   |-- resource_service.py
|   |-- simulation_service.py
|   `-- training_service.py
|-- templates/
|-- static/css/style.css
|-- tests/
|-- datasets/
|   |-- algoguard_big.csv
|   |-- algoguard_bigger.csv
|   `-- algoguard_biggest.csv
|-- database/
|-- uploads/
|-- saved_models/
|-- reports/
`-- research/
```

`research/` is an archived notebook workspace and is not imported by the Flask application.

## Installation on Windows PowerShell

Run all commands from the project root, the directory containing `app.py` and `requirements.txt`:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, the environment can still be used directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py
```

## Database Migration

Migrations are additive and preserve existing rows:

```powershell
python migrate.py
```

The application also applies pending migrations at startup. Do not delete `database/algoguard.sqlite3` to upgrade an existing installation.

## Run the Application

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

Default local account on a new database:

```text
Username: admin
Password: admin123
```

Change these environment values before first startup in a new environment:

```powershell
$env:ALGOGUARD_ADMIN_USERNAME = "your_admin"
$env:ALGOGUARD_ADMIN_PASSWORD = "a-strong-password"
$env:ALGOGUARD_ADMIN_EMAIL = "admin@example.com"
$env:ALGOGUARD_SECRET_KEY = "a-long-random-secret"
python app.py
```

Multiple accounts are supported. Administrators can create Administrator or Analyst accounts from the Admins page. Password hashes and accounts persist in SQLite. Login state uses a signed Flask browser session and ends when the session cookie is cleared or the user logs out; it is not a permanent login token.

## CSV Contract

- Use one `.csv` file per upload.
- Include a header row.
- Put all input features before the target.
- Put the target label in the final column.
- Use exactly two target classes representing Normal and Attack.
- Include at least three rows per class; realistic evaluation requires substantially more.
- Numeric and text features are accepted. Missing feature values are imputed.
- Target values cannot be missing.

Example:

```csv
dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sload,dload,sinpkt,dinpkt,label
2.001348,tcp,-,FIN,16,14,862,802,14.490233,254,252,3233.820312,2977.99292,133.423196,146.368077,Normal
1.665064,tcp,ftp-data,FIN,14,6,8928,320,11.410973,31,29,39835.10547,1282.833618,128.081847,332.462,Normal
0.472012,tcp,http,FIN,10,8,834,354,36.016032,254,252,12728.48926,5254.103516,52.445778,60.35343,Attack
0.320972,tcp,http,FIN,10,8,784,1256,52.964122,62,252,17596.55078,27391.79688,35.663556,38.518145,Attack
```

Each row is one traffic-flow observation. Each feature column is one measured property. The final `label` is the known class used during supervised training. Adding a row labeled `Attack` does not force future inputs with those exact values to be attacks; it gives the models another labeled example from which to learn.

## Included Manual Dataset Files

Three ready-to-upload CSV files are included in `datasets/`:

- `algoguard_big.csv`: 5,000 rows
- `algoguard_bigger.csv`: 10,000 rows
- `algoguard_biggest.csv`: 20,000 rows

The files are nested, stratified, non-replacement samples of the labeled UNSW-NB15 training partition. Each file uses the same 15 input features: `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sinpkt`, and `dinpkt`. The final binary `label` uses `Normal` and `Attack`. Other source columns, including the identifier and `attack_cat`, were removed to keep training practical and prevent identifier noise or direct target leakage.

AlgoGuard does not generate or combine these files at runtime. Open Upload and select each CSV manually, one at a time. Each upload creates a separate run with its own seven model results and recommendation.

Dataset source and attribution: [The UNSW-NB15 Dataset, UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Academic work using these files should cite the dataset publications listed by UNSW.

## Deploy and Predict

Open a completed run, review all seven rows, then select **Deploy recommended model**. Training never replaces the active model automatically. Deployment saves the pipeline, model identity, source run, metric summary, and deployment timestamp to `saved_models/deployed_model.joblib` and records the active deployment in SQLite.

The Prediction page builds its fields from the active artifact's saved feature schema. A prediction stores Normal or Attack, confidence, model name, timestamp, latency, and alert status. Attack creates an alert and audit events; Normal does not create an attack alert.

## System Logs and Alerts

Alert History contains anomalous predictions. System Logs contain authentication, upload, validation, preprocessing, training, ranking, deployment, prediction, reporting, and application events. Logs are newest first and support date, module, status, model, run ID, and text filters.

## Quality Assurance

Install QA tools and run all checks:

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check app.py migrate.py services tests
python -m pytest -q
```

## Troubleshooting

`ModuleNotFoundError`: activate the same virtual environment where dependencies were installed, or invoke `.\venv\Scripts\python.exe` directly.

Invalid CSV: confirm the file is CSV, the target is last, exactly two target classes exist, and both classes have enough rows.

No active deployment: finish a valid training run and explicitly deploy its recommended model.

Missing artifact: the database record and `saved_models/deployed_model.joblib` must agree. Redeploy a valid recommended model from Training Runs.

Forgot local password: do not delete a database containing needed records. For a disposable new local installation only, recreate the database and seed credentials. Production-style password reset administration is outside this prototype's scope.

## Scope

AlgoGuard is intended for academic and controlled local use. Model quality depends on representative, correctly labeled data. It is not a replacement for a production IDS/IPS, live capture agent, or security operations platform.
