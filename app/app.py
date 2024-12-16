import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
import requests
import time
import json
import csv
import pickle
import pandas as pd

from future_sales_prediction_2024 import (
    MainPipeline,
    MemoryReducer,
    DataLoader,
    Trainer,
    FeatureImportanceLayer,
    HyperparameterTuner,
    Explainability,
    ErrorAnalysis,
)

app = FastAPI()

# Data validation schema
class PredictRequest(BaseModel):
    ID: int = Field(..., ge=0)
    shop_id: int = Field(..., ge=0, le=59)
    item_id: int = Field(..., ge=0, le=22169)


class Prediction(BaseModel):
    ID: int
    prediction: float

class InferenceResponse(BaseModel):
    predictions: List[Prediction] = Field(..., description="A list of predictions, each containing an ID and a predicted value")



@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}


@app.post("/infer", response_model=InferenceResponse)
async def run_inference(request: List[PredictRequest]):

    if request is None:
        raise HTTPException(status_code=400, detail="No inference request data found")

    pipeline = MainPipeline(config_path='/config.yaml')
    '''

    test_data = pd.DataFrame([item.dict() for item in request])
    pipeline.save_to_destination(test_data, 'new_test')
    test_path = pipeline.config["local_paths"]['new_test']

    pipeline.run(test_file = test_path)
    '''

    full_featured_data = pipeline.loader.load('full_featured_data')

    trainer = Trainer()

    with open("../artifacts/params/best_params.json", 'r') as file:
        best_params = json.load(file)

    X_train, y_train, X_test = trainer.split_data(full_featured_data)

    y_pred, model, rmse = trainer.train_predict(
        X_train, y_train, X_test, model_name='XGBRegressor', best_params = best_params
    )

    predictions = [{"ID": int(data.ID[i]), "prediction": float(y_pred[i])} for i in range(len(y_pred))]
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
