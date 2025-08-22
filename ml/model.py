import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data

def train_model(X_train, y_train):
    """Train and return a model."""
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    return clf

def inference(model, X):
    """Run model inferences and return predictions."""
    return model.predict(X)

def save_model(obj, path):
    """Save a model or any encoder/label binarizer to disk."""
    joblib.dump(obj, path)

def load_model(path):
    """Load a model or encoder/label binarizer from disk."""
    return joblib.load(path)

def compute_model_metrics(y, preds):
    """Compute precision, recall, and fbeta (beta=1)."""
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta

def performance_on_categorical_slice(
    data, column, value, model, encoder, lb, cat_features
):
    """
    Compute metrics on a slice where `data[column] == value`.
    Returns (precision, recall, fbeta).
    """
    slice_df = data[data[column] == value]
    if slice_df.empty:
        return np.nan, np.nan, np.nan

    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds_slice = inference(model, X_slice)
    return compute_model_metrics(y_slice, preds_slice)

