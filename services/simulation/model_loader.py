import os
import shutil
import zipfile

import joblib


DEFAULT_MODEL_NAME = "Stacking_Top3_LR"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts", "pretrained")
ARTIFACT_DIR = os.path.abspath(
    os.environ.get("ALGOGUARD_PRETRAINED_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
)


class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_name = None

    def _artifact_dirs(self):
        """Return configured model artifact locations."""
        return [ARTIFACT_DIR]

    def _find_artifacts(self, model_name):
        for artifact_dir in self._artifact_dirs():
            pkl_path = os.path.join(artifact_dir, f"{model_name}.pkl")
            zip_path = os.path.join(artifact_dir, f"{model_name}.zip")

            if os.path.exists(pkl_path) or os.path.exists(zip_path):
                return pkl_path, zip_path

        artifact_dir = self._artifact_dirs()[0]
        return os.path.join(artifact_dir, f"{model_name}.pkl"), os.path.join(
            artifact_dir,
            f"{model_name}.zip",
        )

    def _extract_model(self, zip_path, pkl_path, model_name):
        expected_file = f"{model_name}.pkl"

        print(f"Model not found. Extracting {expected_file} from {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            member_name = next(
                (name for name in zip_ref.namelist() if os.path.basename(name) == expected_file),
                None,
            )

            if member_name is None:
                raise FileNotFoundError(f"{expected_file} not found in archive: {zip_path}")

            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with zip_ref.open(member_name) as source, open(pkl_path, "wb") as target:
                shutil.copyfileobj(source, target)

        print(f"Successfully extracted {pkl_path}")

    def load_model(self, model_name=DEFAULT_MODEL_NAME):
        """Load a pretrained model from the configured artifact folders."""
        pkl_path, zip_path = self._find_artifacts(model_name)

        if not os.path.exists(pkl_path) and os.path.exists(zip_path):
            self._extract_model(zip_path, pkl_path, model_name)

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model file not found: {pkl_path}")

        self.model = joblib.load(pkl_path)
        self.model_name = model_name
        print(f"Model loaded successfully: {self.model_name}")
        return self.model
