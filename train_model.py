# train_model.py
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    save_model,
    load_model,
    compute_model_metrics,
    performance_on_categorical_slice,
)

# ---- Paths ----
project_path = str(Path(__file__).resolve().parent)
data_path = os.path.join(project_path, "data", "census.csv")
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")
slice_path = os.path.join(project_path, "slice_output.txt")

print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# ---- Split train/test ----
train, test = train_test_split(
    data, test_size=0.20, random_state=42, stratify=data["salary"]
)

# ---- Categorical features (DO NOT MODIFY) ----
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# ---- Process data ----
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
    encoder=None,
    lb=None,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# ---- Train & persist model/encoder ----
model = train_model(X_train, y_train)
save_model(model, model_path)
save_model(encoder, encoder_path)
print("Model saved to", model_path)
print("Encoder saved to", encoder_path)

# ---- (Optional) reload to mimic real usage ----
model = load_model(model_path)

# ---- Inference & overall metrics ----
preds = inference(model, X_test)
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# ---- Per-slice metrics ----
# clear previous slice output
open(slice_path, "w").close()

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test.loc[test[col] == slicevalue].shape[0]
        sp, sr, sfb = performance_on_categorical_slice(
            test,           # full test DataFrame
            col,            # column to slice on
            slicevalue,     # specific value within the column
            model,          # trained model
            encoder,        # fitted encoder from training
            lb,             # fitted label binarizer from training
            cat_features,   # list of categorical features
        )
        with open(slice_path, "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {sp:.4f} | Recall: {sr:.4f} | F1: {sfb:.4f}", file=f)

print(f"Per-slice metrics written to {slice_path}")