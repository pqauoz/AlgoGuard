import os

from flask import Flask, jsonify, render_template, request

from feature_extractor import FeatureExtractor
from flow_builder import FlowBuilder
from model_loader import ModelLoader
from predictor import Predictor
from preprocessor import Preprocessor


app = Flask(__name__)

print("Initializing AlgoGuard System...")

preprocessor = Preprocessor()
preprocessor.feature_extractor = FeatureExtractor()

model_loader = ModelLoader()
model = model_loader.load_model("Stacking_Top3_GB")

predictor = Predictor(model, preprocessor)
flow_builder = FlowBuilder()

print("AlgoGuard System Ready!\n")


@app.route("/")
def index():
    """Render the legacy manual prediction dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for real-time simulation predictions."""
    try:
        raw_json = request.json or {}
        flow_data = flow_builder.build_from_json(raw_json)
        prediction_label, confidence = predictor.predict(flow_data)

        result = {
            "status": "success",
            "prediction": "Attack" if prediction_label == 1 else "Normal",
            "confidence": round(confidence * 100, 2),
            "flow_data": flow_data,
        }

        return jsonify(result)

    except Exception as error:
        return jsonify({"status": "error", "message": str(error)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("ALGOGUARD_LEGACY_PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
