import os
import time

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from services.evaluation_service import calculate_metrics, identify_best_model
from services.preprocessing_service import build_feature_preprocessor


RANDOM_STATE = 42


def _safe_model_filename(model_name):
    """Convert a display name into a stable Joblib filename."""
    return model_name.lower().replace(" ", "_") + ".joblib"


def _build_model_specs():
    """Create the ensemble models compared by AlgoGuard."""
    random_forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gradient_boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)

    voting_classifier = VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=80,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        voting="soft",
    )

    return {
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
        "Voting Classifier": voting_classifier,
    }


def train_and_compare_models(prepared_dataset, saved_model_folder):
    """Train every model, save each pipeline, and return comparison results."""
    os.makedirs(saved_model_folder, exist_ok=True)
    model_results = []

    for model_name, estimator in _build_model_specs().items():
        preprocessor = build_feature_preprocessor(
            prepared_dataset.numeric_columns,
            prepared_dataset.categorical_columns,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        start_time = time.perf_counter()
        pipeline.fit(prepared_dataset.X_train, prepared_dataset.y_train)
        predictions = pipeline.predict(prepared_dataset.X_test)
        processing_time = time.perf_counter() - start_time

        model_path = os.path.join(saved_model_folder, _safe_model_filename(model_name))
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_columns": prepared_dataset.feature_columns,
                "target_column": prepared_dataset.target_column,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            model_path,
        )

        metrics = calculate_metrics(model_name, prepared_dataset.y_test, predictions, processing_time)
        metrics["model_path"] = model_path
        metrics["model_file"] = os.path.basename(model_path)
        model_results.append(metrics)

    best_model = identify_best_model(model_results)

    return {
        "model_results": model_results,
        "best_model": best_model,
    }
