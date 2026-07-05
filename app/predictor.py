import numpy as np

class Predictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def predict(self, raw_flow_data):
        """
        Takes raw JSON data, processes it, and returns the prediction.
        """
        # 1. Extract 10 raw features into a DataFrame becuase i have 10  as raw
        df_10 = self.preprocessor.feature_extractor.extract(raw_flow_data)
        
        # 2. Transform 10 features -> 145 encoded & scaled features
        features_145 = self.preprocessor.transform(df_10)
        
        # 3. Make prediction
        prediction = self.model.predict(features_145)[0]
        
        # 4. Get confidence score
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_145)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 1.0
        
        return int(prediction), confidence