import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# ---- paths to serialized artifacts (adjust if you saved them elsewhere) ----
encoder_path = "model/encoder.pkl"   # TODO: update if your path differs
encoder = load_model(encoder_path)

model_path = "model/model.pkl"       # TODO: update if your path differs
model = load_model(model_path)

# ---- FastAPI app ----
app = FastAPI(title="Census Income Inference API")

# ---- Root GET ----
@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Welcome to the Census Income inference API."}

# ---- Inference POST ----
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

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

    # preprocess with the provided function
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None,
    )

    # model inference
    preds = inference(model, X)

    # map numeric prediction to label string
    labeled = apply_label(preds)
    # return the first (and only) prediction for this single-row request
    return {"result": labeled[0] if hasattr(labeled, "__len__") else labeled}