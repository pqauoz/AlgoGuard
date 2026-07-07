# AlgoGuard

*AlgoGuard* is a lightweight machine learning-based network anomaly detection framework designed for *resource-constrained organizations*. It provides an easy-to-use web dashboard for analyzing network traffic and detecting anomalous behavior using a stacked ensemble machine learning model.

---

# Features

- Lightweight and easy to deploy
- Machine learning-based anomaly detection
- Flask web dashboard
- Prediction using a stacked ensemble model
- Automatic model extraction on first run
- Designed for small organizations with limited computing resources

---

# System Requirements

Before installing AlgoGuard, make sure your computer has the following software installed:

- *Python 3.8 or higher*
  https://www.python.org/downloads/

- *Git*
  https://git-scm.com/downloads

---

# Installation

## 1. Clone the Repository

Open Command Prompt or Terminal and run:

git clone https://github.com/pqauoz/AlgoGuard.git
cd AlgoGuard

---

## 2. Create a Virtual Environment

### Windows

python -m venv venv
venv\Scripts\activate

---

## 3. Install the Required Packages

Install all required Python libraries:

pip install flask scikit-learn pandas numpy joblib

---

# Running the Application

Navigate to the application folder:

cd app

Start the Flask server:

python app.py

During the first execution, AlgoGuard will automatically extract the compressed machine learning model. This process only happens once and usually takes around *5 seconds or maybe longer*.

After the server starts, open your browser and visit:

http://localhost:5000

---

# Using AlgoGuard

The web dashboard allows users to manually input network traffic features and analyze whether the traffic is normal or anomalous.

### Test Normal Traffic

Input:

| Feature | Value |
|---------|------|
| Duration | 0.001 |
| Rate | 15 |
| Protocol | tcp |
| State | CON |

*Expected Result*

Normal Traffic

---

### Test Attack Traffic

Input:

| Feature | Value |
|---------|------|
| Duration | 0.000001 |
| Rate | 5000 |
| Protocol | tcp |
| State | REQ |

*Expected Result*
Attack Detected


---

# Project Structure

AlgoGuard/
│
├── app/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   ├── models/
│   └── Stacking_Top3_GB.zip
│
├── dataset/
│
├── training/
│
├── README.md
│
└── requirements.txt

---

# Machine Learning Model

AlgoGuard uses a *Stacked Ensemble Learning* approach consisting of:

### Base Learners

- Random Forest
- AdaBoost
- Gradient Boosting

### Meta Learner

- Gradient Boosting

This ensemble combines predictions from multiple models to improve anomaly detection accuracy while maintaining lightweight performance suitable for deployment on low-resource systems.

---

# Troubleshooting

## ModuleNotFoundError

If you receive an error similar to:

text
ModuleNotFoundError

Activate the virtual environment first:

### Windows

venv\Scripts\activate

Then reinstall the required packages:

pip install flask scikit-learn pandas numpy joblib

---

## Port 5000 Already in Use

If Flask reports that port *5000* is already being used:

Open:

app/app.py

Find:

port=5000

Change it to:

port=5001

Run the application again and open:

http://localhost:5001

---

## Model File Not Found

If the application cannot find the machine learning model:

- Verify that *Stacking_Top3_GB.zip* is located inside the *app/* directory.
- Restart the application.
- The model will automatically be extracted during startup.

---

# Developers

Developed as part of a research project (currenlty just the simulation):

*AlgoGuard: A Lightweight Framework for Local Network Anomaly Detection in Resource-Constrained Organizations*

---

# License

This project is intended for *academic and research purposes*.
