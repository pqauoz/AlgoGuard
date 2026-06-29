import joblib
import pandas as pd


def predict(features: pd.DataFrame):

    # Load trained model
    model = joblib.load("models/stacking.pkl")

    # Load scaler used during training
    scaler = joblib.load("models/scaler.pkl")

    # Scale features
    X = scaler.transform(features)

    # Predict
    predictions = model.predict(X)

    # Prediction probabilities (if supported)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
    else:
        probabilities = None

    return predictions, probabilities