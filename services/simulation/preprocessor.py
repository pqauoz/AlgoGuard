import os

import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts", "pretrained")
ARTIFACT_DIR = os.path.abspath(
    os.environ.get("ALGOGUARD_PRETRAINED_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
)


class Preprocessor:
    def __init__(self):
        self.scaler = None
        self.encoded_columns = None
        self.load_resources()

    def load_resources(self):
        """Load the scaler and encoded feature column names."""
        scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
        columns_path = os.path.join(ARTIFACT_DIR, "encoded_columns.npy")

        self.scaler = joblib.load(scaler_path)
        if hasattr(self.scaler, "feature_names_in_"):
            self.encoded_columns = self.scaler.feature_names_in_
        else:
            self.encoded_columns = np.load(columns_path, allow_pickle=True)

        print(f"Preprocessor loaded with {len(self.encoded_columns)} encoded columns.")

    def transform(self, df_10_features):
        """Transform 10 raw features into the encoded and scaled model input."""
        if self.scaler is None or self.encoded_columns is None:
            raise ValueError("Preprocessor resources not loaded.")

        df_encoded = pd.get_dummies(df_10_features, columns=["proto", "state"], drop_first=True)
        df_encoded = df_encoded.reindex(columns=self.encoded_columns, fill_value=0)

        return self.scaler.transform(df_encoded)
