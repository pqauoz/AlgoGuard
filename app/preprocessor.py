import joblib
import numpy as np
import pandas as pd
import os

class Preprocessor:
    def __init__(self):
        self.scaler = None
        self.encoded_columns = None
        self.load_resources()
    
    def load_resources(self):
        """Load the scaler and the 145 column names from the app directory."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the StandardScaler
        scaler_path = os.path.join(base_dir, "scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        columns_path = os.path.join(base_dir, "encoded_columns.npy")
        self.encoded_columns = np.load(columns_path, allow_pickle=True)
        
        print(f"✓ Preprocessor loaded. Scaler and {len(self.encoded_columns)} encoded columns ready.")

    def transform(self, df_10_features):
        """
        Transform 10 raw features into 145 encoded and scaled features.
        """
        if self.scaler is None or self.encoded_columns is None:
            raise ValueError("Preprocessor resources not loaded.")
        
        df_encoded = pd.get_dummies(df_10_features, columns=['proto', 'state'], drop_first=True)
        
        df_encoded = df_encoded.reindex(columns=self.encoded_columns, fill_value=0)

        features_scaled = self.scaler.transform(df_encoded)
        
        return features_scaled