class Predictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, raw_flow_data):
        """Process raw flow data and return the prediction label plus confidence."""
        df_10 = self.preprocessor.feature_extractor.extract(raw_flow_data)
        features_145 = self.preprocessor.transform(df_10)
        prediction = self.model.predict(features_145)[0]

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features_145)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 1.0

        return int(prediction), confidence
