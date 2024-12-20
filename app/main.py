from typing import Union

from fastapi import FastAPI
import dill
from pydantic import BaseModel
import pandas as pd
import numpy as np

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import joblib
app = FastAPI()

import dill
with open('./app/logr_v1.pkl', 'rb') as f:
    reloaded_model = dill.load(f)


class Payload(BaseModel):
    gender: str
    ever_married: str
    age: float
    graduated: str
    profession: str
    work_experience: float
    spending_score: str
    family_size:float
    var_1: float

app = FastAPI()


@app.get("/")
def read_root():
    return {
        "Name": "Jyothismaria Joseph",
        "Project": "Customer Segmentation",
        "Model": "L ogistiv Regression"
    }


@app.post("/predict")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    y_hat = reloaded_model.predict(df)
    y_hat_result = int(y_hat[0]) if isinstance(y_hat[0], np.int64) else y_hat[0]
    
    # Return the prediction as a serializable response
    return {"prediction": y_hat_result}