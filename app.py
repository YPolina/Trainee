from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import os
import subprocess
import numpy as np
from future_sales_prediction_2024.core.model_training import train_predict, data_split
from future_sales_prediction_2024.core.hyperparameters import hyperparameter_tuning
from xgboost import XGBRegressor
from google.cloud import storage
from typing import List, Dict, Union

app = FastAPI()

if os.path.exists("data_pulled/full_featured_data.csv"):
    DATA_PATH = "data_pulled/full_featured_data.csv"
else:
    try:
        print("Data is not found. Data loading from Google Cloud")
        local_path = "./data_pulled"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client = storage.Client()
        # Access the bucket and download the data
        bucket = client.get_bucket('my-dvc-bucket_future_prections')
        blob = bucket.blob('preprocessed_data/full_featured_data.csv')
        blob.download_to_filename('/data_pulled/full_featured_data.csv')
    except Exception as e:
        raise RuntimeError(f"Error during data loading: {e}")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in loading data {e}")


# Request Validation
class InferenceRequest(BaseModel):
    model_type: str
    model_params: dict = None
    tune_hyperparameters: bool = False
    param_space: dict = None

class InferenceResponse(BaseModel):
    predictions: List[float]
    model_params: Union[Dict, None]

user_request: InferenceRequest = None


@app.get("/")
async def root():
    return {"message": "Welcome to the Future Sales Prediction API"}

@app.post('/submit_request')
async def submit_request(request: InferenceRequest = Body(...)):
    """
    Endpoint to accept `InferenceRequest` data
    Stores the request data in memory and redirects to `/infer`
    """
    global user_request
    user_request = request

    return RedirectResponse(url="/infer", status_code=303)



@app.get("/infer", response_model=InferenceResponse)
async def run_inference():

    global user_request, DATA_PATH
    if user_request is None:
        raise HTTPException(status_code=400, detail="No inference request data found")

    try:
        data = load_data(DATA_PATH)

        X_train, y_train, X_test = data_split(data)

        if user_request.model_type.lower() == "xgboost":
            from xgboost import XGBRegressor

            model_class = XGBRegressor
        elif user_request.model_type.lower() == "lightgbm":
            from lightgbm import LGBMRegressor

            model_class = LGBMRegressor
        elif user_request.model_type.lower() == "linearregression":
            from sklearn.linear_model import LinearRegression

            model_class = LinearRegression
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")

        if (user_request.tune_hyperparameters) and (
            user_request.model_type.lower() != "linear_regression"
        ):
            param_space = user_request.param_space or {
                "max_depth": hp.randint("max_depth", 3, 11),
                "eta": hp.uniform("eta", 0.01, 0.3),
                "gamma": hp.uniform("gamma", 0, 5),
            }
            best_params = hyperparameter_tuning(
                X=X_train, y=y_train, model_class=model_class, param_space=param_space
            )
            model_params = best_params
        else:
            model_params = user_request.model_params

        model = model_class()
        y_pred, trained_model = train_predict(
            X_train, y_train, X_test, model, model_params
        )

        return InferenceResponse(
            predictions=y_pred.tolist(),
            model_params=model_params,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("future_sales_prediction_2024.api.app:app", host="127.0.0.1", port=8000, reload=True)
