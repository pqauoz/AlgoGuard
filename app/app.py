from flask import Flask, render_template, request, jsonify
from model_loader import ModelLoader
from preprocessor import Preprocessor
from predictor import Predictor
from feature_extractor import FeatureExtractor
from flow_builder import FlowBuilder

app = Flask(__name__)


print("Initializing AlgoGuard System...")


preprocessor = Preprocessor()


preprocessor.feature_extractor = FeatureExtractor()

model_loader = ModelLoader()
model = model_loader.load_model("Stacking_Top3_GB.pkl") 

predictor = Predictor(model, preprocessor)


flow_builder = FlowBuilder()

print("✓ AlgoGuard System Ready!\n")

# --- Flask Routes ---

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for real-time predictions."""
    try:
        raw_json = request.json
        
        flow_data = flow_builder.build_from_json(raw_json)
        
        prediction_label, confidence = predictor.predict(flow_data)

        result = {
            'status': 'success',
            'prediction': 'Attack' if prediction_label == 1 else 'Normal',
            'confidence': round(confidence * 100, 2),
            'flow_data': flow_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)