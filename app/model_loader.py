import joblib
import os
import zipfile

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_name = None
    
    def load_model(self, model_name="Stacking_Top3_GB"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_dir, f"{model_name}.pkl")
        zip_path = os.path.join(base_dir, f"{model_name}.zip")
        
        # If the .pkl doesn't exist but the .zip does, extract it eyyy nangdaya
        if not os.path.exists(pkl_path) and os.path.exists(zip_path):
            print(f" Model not found. Extracting {model_name}.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            print(f" Successfully extracted {model_name}.pkl")
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model file not found: {pkl_path}")
            
        self.model = joblib.load(pkl_path)
        self.model_name = model_name
        print(f"✓ Model loaded successfully: {self.model_name}")
        return self.model