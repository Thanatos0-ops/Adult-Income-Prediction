from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from contextlib import asynccontextmanager

# Gobal variable to cache our trained pipeline model
MODEL_PATH = "models/artifacts/income_classifier_pipeline.joblib"
model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """load the model pipeline binary into memory when the server boots up"""
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f'Model Artifact binary not found at {MODEL_PATH}')
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model Pipeline loaded successfully into memory")

    yield

# Initialize FastAPI
app = FastAPI(
    title="Adult Income Prediction Service",
    version="1.0.0",
    lifespan= lifespan
)

# Define the Expected Inputs schema using Pydantic

class IncomePredictionInput(BaseModel):
    age: int
    workclass: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int  
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def health_check():
    return {
        "status" : "healthy", 
        "model_loaded": model_pipeline is not None
        }

@app.post("/predict")
def predict_income(payload: IncomePredictionInput):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")
    
    try:
        # Convert incoming payload to a dictionary
        input_data = payload.model_dump()

        # Map Pythonic snake_case keys back to original dot notation features expected by the pipeline
        mapped_data = {
            'age': input_data['age'],
            'workclass': input_data['workclass'],
            'education.num': input_data['education_num'],
            'marital.status': input_data['marital_status'],
            'occupation': input_data['occupation'],
            'relationship': input_data['relationship'],
            'race': input_data['race'],
            'sex': input_data['sex'],
            'capital.gain': input_data['capital_gain'],
            'capital.loss': input_data['capital_loss'],
            'hours.per.week': input_data['hours_per_week'],
            'native.country': input_data['native_country']            
        }

        # Convert the single prediction request into a 1 row pandas dataframe
        df_input = pd.DataFrame([mapped_data])

        # Execute the full pipeline prediction
        prediction = int(model_pipeline.predict(df_input)[0])
        probability = float(model_pipeline.predict_proba(df_input)[0][1])

        return {
            'income_bracket': ">50K" if prediction == 1 else "<=50K",
            "prediction_code": prediction,
            "probability_of_high_income": round(probability, 4)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Failed: {str(e)}")
    
