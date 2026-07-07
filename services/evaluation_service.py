import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


NORMAL_LABELS = {"0", "normal", "benign", "legitimate", "clean"}


def infer_normal_label(labels):
    """Find the normal label using common network anomaly naming patterns."""
    label_values = [str(label).strip() for label in labels]
    for label in label_values:
        if label.lower() in NORMAL_LABELS:
            return label
    return sorted(set(label_values))[0]


def false_positive_rate(y_true, y_pred):
    """Calculate FPR by treating non-normal labels as anomalies."""
    normal_label = infer_normal_label(y_true)
    true_anomaly = np.array([str(label).strip() != normal_label for label in y_true])
    predicted_anomaly = np.array([str(label).strip() != normal_label for label in y_pred])

    tn, fp, _, _ = confusion_matrix(true_anomaly, predicted_anomaly, labels=[False, True]).ravel()
    denominator = fp + tn
    if denominator == 0:
        return 0.0

    return fp / denominator


def prediction_traffic_counts(y_pred):
    """Count predicted normal and anomalous traffic records."""
    normal_label = infer_normal_label(y_pred)
    normal_count = sum(str(label).strip() == normal_label for label in y_pred)
    anomaly_count = len(y_pred) - normal_count

    return {
        "predicted_normal_count": int(normal_count),
        "predicted_anomaly_count": int(anomaly_count),
    }


def calculate_metrics(model_name, y_true, y_pred, processing_time):
    """Return the model metrics shown in the comparison table."""
    traffic_counts = prediction_traffic_counts(y_pred)

    return {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "false_positive_rate": round(false_positive_rate(y_true, y_pred) * 100, 2),
        "processing_time": round(processing_time, 4),
        **traffic_counts,
    }


def identify_best_model(model_results):
    """Choose the best model by F1-score, accuracy, FPR, and processing time."""
    return sorted(
        model_results,
        key=lambda result: (
            -result["f1_score"],
            -result["accuracy"],
            result["false_positive_rate"],
            result["processing_time"],
        ),
    )[0]
