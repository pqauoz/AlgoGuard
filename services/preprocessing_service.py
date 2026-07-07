from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NORMAL_LABELS = {"0", "normal", "benign", "legitimate", "clean"}


@dataclass
class PreparedDataset:
    """Container for cleaned data and metadata needed by training services."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list
    target_column: str
    numeric_columns: list
    categorical_columns: list
    summary: dict


def _make_one_hot_encoder():
    """Create a version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_feature_preprocessor(numeric_columns, categorical_columns):
    """Build the encoder and scaler used by every model pipeline."""
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _normal_anomaly_counts(labels):
    """Count normal and anomalous rows using common IDS label conventions."""
    normalized = labels.astype(str).str.strip().str.lower()
    normal_mask = normalized.isin(NORMAL_LABELS)

    return {
        "normal_count": int(normal_mask.sum()),
        "anomaly_count": int((~normal_mask).sum()),
    }


def _validate_dataframe(df):
    """Validate that a CSV has enough data for supervised classification."""
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    df = df.dropna(how="all").dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("The dataset must contain feature columns and one target label column.")

    if df.shape[0] < 4:
        raise ValueError("The dataset needs at least four rows for training and testing.")

    return df


def prepare_dataset(file_path, test_size=0.25, random_state=42):
    """Load a CSV, validate it, split features and labels, and prepare metadata."""
    try:
        df = pd.read_csv(file_path)
    except Exception as error:
        raise ValueError(f"Unable to read the CSV file. Details: {error}") from error

    df = _validate_dataframe(df)
    target_column = df.columns[-1]

    if not str(target_column).strip():
        raise ValueError("The last column must be a named target label column.")

    raw_target = df[target_column]
    if raw_target.isna().any() or raw_target.astype(str).str.strip().eq("").any():
        raise ValueError("The target label column contains missing values.")

    X = df.iloc[:, :-1].copy()
    y = raw_target.astype(str).str.strip()
    X = X.replace([np.inf, -np.inf], np.nan)

    class_counts = y.value_counts()
    if class_counts.size < 2:
        raise ValueError("The target label column must contain at least two classes.")

    if class_counts.min() < 2:
        raise ValueError("Each target class needs at least two rows for a reliable train/test split.")

    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    test_rows = max(int(np.ceil(df.shape[0] * test_size)), int(class_counts.size))
    max_test_rows = df.shape[0] - int(class_counts.size)
    if test_rows > max_test_rows:
        raise ValueError("The dataset is too small to keep every class in both training and testing sets.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_rows,
        random_state=random_state,
        stratify=y,
    )

    label_counts = class_counts.to_dict()
    traffic_counts = _normal_anomaly_counts(y)

    summary = {
        "rows": int(df.shape[0]),
        "features": int(X.shape[1]),
        "target_column": str(target_column),
        "class_count": int(class_counts.size),
        "label_counts": {str(label): int(count) for label, count in label_counts.items()},
        "normal_count": traffic_counts["normal_count"],
        "anomaly_count": traffic_counts["anomaly_count"],
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "numeric_features": len(numeric_columns),
        "categorical_features": len(categorical_columns),
    }

    return PreparedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=X.columns.tolist(),
        target_column=str(target_column),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        summary=summary,
    )
