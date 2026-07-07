import importlib.util
import os
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LEGACY_APP_DIR = os.path.join(BASE_DIR, "legacy_simulation")

_simulation_runtime = None


class SimulationServiceError(RuntimeError):
    """Raised when the pretrained simulation model cannot run."""


def _load_legacy_class(module_name, filename, class_name):
    """Load a class from the legacy prototype folder without importing app.py."""
    module_path = os.path.join(LEGACY_APP_DIR, filename)
    if not os.path.exists(module_path):
        raise SimulationServiceError(f"Missing legacy simulation file: {filename}")

    unique_name = f"algoguard_legacy_{module_name}"
    if unique_name in sys.modules:
        module = sys.modules[unique_name]
    else:
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec is None or spec.loader is None:
            raise SimulationServiceError(f"Unable to load simulation module: {filename}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)

    try:
        return getattr(module, class_name)
    except AttributeError as error:
        raise SimulationServiceError(f"Missing {class_name} in {filename}") from error


class SimulationService:
    """Wrap the older manual prediction prototype for use in the root Flask app."""

    def __init__(self):
        ModelLoader = _load_legacy_class("model_loader", "model_loader.py", "ModelLoader")
        Preprocessor = _load_legacy_class("preprocessor", "preprocessor.py", "Preprocessor")
        Predictor = _load_legacy_class("predictor", "predictor.py", "Predictor")
        FeatureExtractor = _load_legacy_class("feature_extractor", "feature_extractor.py", "FeatureExtractor")
        FlowBuilder = _load_legacy_class("flow_builder", "flow_builder.py", "FlowBuilder")

        try:
            preprocessor = Preprocessor()
            preprocessor.feature_extractor = FeatureExtractor()
            model = ModelLoader().load_model("Stacking_Top3_GB")
        except Exception as error:
            raise SimulationServiceError(f"Unable to load pretrained simulation model: {error}") from error

        self.predictor = Predictor(model, preprocessor)
        self.flow_builder = FlowBuilder()

    def predict(self, payload):
        """Build a flow from user input and return the model prediction."""
        flow_data = self.flow_builder.build_from_json(payload or {})
        prediction_label, confidence = self.predictor.predict(flow_data)
        prediction = "Attack" if prediction_label == 1 else "Normal"

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "flow_data": flow_data,
        }


def run_simulation(payload):
    """Run the singleton simulation service."""
    global _simulation_runtime

    if _simulation_runtime is None:
        _simulation_runtime = SimulationService()

    return _simulation_runtime.predict(payload)
