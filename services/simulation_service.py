import os

from services.simulation.feature_extractor import FeatureExtractor
from services.simulation.flow_builder import FlowBuilder
from services.simulation.model_loader import ModelLoader
from services.simulation.predictor import Predictor
from services.simulation.preprocessor import Preprocessor


SIMULATION_MODEL_NAME = os.environ.get("ALGOGUARD_SIMULATION_MODEL", "Stacking_Top3_LR")

_simulation_runtime = None


class SimulationServiceError(RuntimeError):
    """Raised when the pretrained simulation model cannot run."""


class SimulationService:
    """Run manual flow detection with the pretrained simulation engine."""

    def __init__(self):
        try:
            preprocessor = Preprocessor()
            preprocessor.feature_extractor = FeatureExtractor()
            model = ModelLoader().load_model(SIMULATION_MODEL_NAME)
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
