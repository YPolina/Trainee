import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import requests
import time

AIRFLOW_BASE_URL = "http://airflow_webserver:8080/api/v1/dags"
AIRFLOW_USERNAME = "admin"
AIRFLOW_PASSWORD = "admin"
app = FastAPI()


# Data validation schema
class TestData(BaseModel):
    ID: int = Field(..., ge=0)
    shop_id: int = Field(..., ge=0, le=59)
    item_id: int = Field(..., ge=0, le=22169)


class PredictRequest(BaseModel):
    test_data: List[TestData]
    model_name: str = Union["XGBRegressor", "RandomForestRegressor", "LinearRegression", "LightGBM"]




class Prediction(BaseModel):
    ID: int = Field(..., ge=0, description="The ID of the test sample, must be a non-negative integer")
    prediction: float = Field(..., description="The predicted value for the given test sample")


class InferenceResponse(BaseModel):
    predictions: List[Prediction] = Field(..., description="A list of predictions, each containing an ID and a predicted value")


@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}



@app.post("/predict", response_model=InferenceResponse)
async def predict(request: PredictRequest):
    """
    Predict endpoint that can optionally train a new model based on the specified model_name

    Parameters:
    - request: PredictRequest - The input test data and specified model class for training a new model

    Returns:
    - InferenceResponse with predictions
    """
    # Extract test data and model_name from the request

    dag_id = "fastapi_training_pipeline"
    url = f"{AIRFLOW_BASE_URL}/{dag_id}/dagRuns"

    payload = {
        "conf": {
            "test_data": [data.dict() for data in request.test_data],
            "model_name": request.model_name,
        }
    }

    response = requests.post(
        url, json=payload, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to trigger DAG {dag_id}. Response: {response.text}",
        )

    dag_run_id = response.json().get("dag_run_id")
    if not dag_run_id:
        raise HTTPException(
            status_code=500,
            detail="Failed to get DAG run ID from Airflow response",
        )

    dag_status_url = f"{AIRFLOW_BASE_URL}/{dag_id}/dagRuns/{dag_run_id}"
    while True:
        status_response = requests.get(dag_status_url, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
        if status_response.status_code != 200:
            raise HTTPException(
                status_code=status_response.status_code,
                detail="Failed to get DAG run status",
            )

        status = status_response.json().get("state")
        if status == "success":
            break
        elif status == "failed":
            raise HTTPException(status_code=500, detail="DAG run failed")

        time.sleep(5)

    predictions_file = "./data/predictions.json"
    try:
        with open(predictions_file, "r") as f:
            predictions = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Predictions file not found")

    return InferenceResponse(predictions=predictions)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
