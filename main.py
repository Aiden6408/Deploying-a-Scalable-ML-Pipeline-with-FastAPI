import os
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# ---------------------------
# Pydantic request schema
# ---------------------------
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# ---------------------------
# Load serialized artifacts
# ---------------------------
ROOT = Path(__file__).resolve().parent
encoder_path = ROOT / "model" / "encoder.pkl"
model_path   = ROOT / "model" / "model.pkl"

encoder = load_model(str(encoder_path))
model   = load_model(str(model_path))

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Census Income Inference API")

@app.get("/")
async def get_root():
    return {"message": "Welcome to the Census Income inference API."}

@app.post("/data/")
async def post_inference(data: Data):
    # to dict, convert aliases with hyphens back to column names
    data_dict = data.dict()
    row = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(row)

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

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None,
    )

    preds = inference(model, X)
    labeled = apply_label(preds)  # could be a scalar string or 1D array/series

    # robustly extract the single prediction value
    if isinstance(labeled, (list, tuple, np.ndarray, pd.Series)):
        value = str(labeled[0])
    else:
        value = str(labeled)

    return {"result": value}