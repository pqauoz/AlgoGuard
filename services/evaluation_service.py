import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

    return float(fp / denominator)


def prediction_traffic_counts(y_pred):
    """Count predicted normal and anomalous traffic records."""
    normal_label = infer_normal_label(y_pred)
    normal_count = sum(str(label).strip() == normal_label for label in y_pred)
    anomaly_count = len(y_pred) - normal_count

    return {
        "predicted_normal_count": int(normal_count),
        "predicted_anomaly_count": int(anomaly_count),
    }


def roc_auc_percent(y_true, prediction_scores, class_labels):
    """Calculate ROC-AUC for binary or multiclass classifier probabilities."""
    if prediction_scores is None or class_labels is None:
        return None

    class_labels = list(class_labels)
    if len(class_labels) < 2:
        return None

    prediction_scores = np.asarray(prediction_scores)
    y_true_array = np.array([str(label).strip() for label in y_true])
    class_array = np.array([str(label).strip() for label in class_labels])

    try:
        if len(class_array) == 2:
            normal_label = infer_normal_label(class_array)
            positive_label = next(label for label in class_array if label != normal_label)
            positive_index = int(np.where(class_array == positive_label)[0][0])
            y_binary = y_true_array == positive_label
            score = roc_auc_score(y_binary, prediction_scores[:, positive_index])
        else:
            score = roc_auc_score(
                y_true_array,
                prediction_scores,
                labels=class_array,
                multi_class="ovr",
                average="weighted",
            )
    except (IndexError, StopIteration, ValueError):
        return None

    return round(score * 100, 2)


def calculate_metrics(
    model_name,
    y_true,
    y_pred,
    processing_time,
    prediction_scores=None,
    class_labels=None,
    cpu_usage=None,
    ram_usage=None,
):
    """Return the model metrics shown in the comparison table."""
    traffic_counts = prediction_traffic_counts(y_pred)

    return {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "roc_auc": roc_auc_percent(y_true, prediction_scores, class_labels),
        "false_positive_rate": round(false_positive_rate(y_true, y_pred) * 100, 2),
        "cpu_usage": round(cpu_usage, 2) if cpu_usage is not None else None,
        "ram_usage": round(ram_usage, 2) if ram_usage is not None else None,
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
