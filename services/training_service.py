import os
import time
import tracemalloc

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

        tracemalloc.start()
        start_time = time.perf_counter()
        start_cpu = time.process_time()
        pipeline.fit(prepared_dataset.X_train, prepared_dataset.y_train)
        predictions = pipeline.predict(prepared_dataset.X_test)
        prediction_scores = pipeline.predict_proba(prepared_dataset.X_test)
        cpu_time = time.process_time() - start_cpu
        processing_time = time.perf_counter() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_usage = (cpu_time / processing_time) * 100 if processing_time else 0
        ram_usage = peak_memory / (1024 * 1024)

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
        model_size = os.path.getsize(model_path) / (1024 * 1024)

        metrics = calculate_metrics(
            model_name,
            prepared_dataset.y_test,
            predictions,
            processing_time,
            prediction_scores=prediction_scores,
            class_labels=pipeline.classes_,
            cpu_usage=cpu_usage,
            ram_usage=ram_usage,
        )
        metrics["model_path"] = model_path
        metrics["model_file"] = os.path.basename(model_path)
        metrics["model_size"] = round(model_size, 2)
        model_results.append(metrics)

    best_model = identify_best_model(model_results)

    return {
        "model_results": model_results,
        "best_model": best_model,
    }
