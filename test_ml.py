from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# -------------------------------------------------------------------
# Use relative paths: project root is the parent of "tests/"
# -------------------------------------------------------------------
PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data" / "census.csv"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def prepared_data():
    """Load census data, split, and process into train/test matrices."""
    assert DATA_PATH.exists(), f"Cannot find census.csv at: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)

    # keep tests lightweight
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=0)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test


def test_model_type(prepared_data):
    """Trained model should be a scikit-learn classifier with predict method."""
    X_train, y_train, _, _ = prepared_data
    model = train_model(X_train, y_train)
    assert isinstance(model, ClassifierMixin)
    assert hasattr(model, "predict")


def test_inference_shape_and_binary_output(prepared_data):
    """Inference should return 1D array of {0,1} predictions of correct length."""
    X_train, y_train, X_test, _ = prepared_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_metrics_within_bounds(prepared_data):
    """Precision, recall, F1 should be finite floats in [0, 1]."""
    X_train, y_train, X_test, y_test = prepared_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    p, r, f1 = compute_model_metrics(y_test, preds)

    for m in (p, r, f1):
        assert isinstance(m, float)
        assert np.isfinite(m)
        assert 0.0 <= m <= 1.0