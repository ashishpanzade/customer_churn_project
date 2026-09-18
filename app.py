import os
import sys
from typing import Literal, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  
from model.feature_engineering import ChurnFeatureEngineer
sys.modules["__main__"].ChurnFeatureEngineer = ChurnFeatureEngineer
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "my_churn_model.pkl")

app = FastAPI(title="Telco Customer Churn Prediction API")
model_pipeline = joblib.load(MODEL_PATH)


class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Union[str, float]

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_must_be_numeric(cls, value):
        try:
            float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'TotalCharges' must be numeric, got {value!r}.")
        return value


class PredictionResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        customer_df = pd.DataFrame([customer.model_dump()])
        prediction = model_pipeline.predict(customer_df)[0]
        churn_probability = model_pipeline.predict_proba(customer_df)[0][1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to score input: {exc}") from exc

    return PredictionResponse(
        prediction="Yes" if prediction == 1 else "No",
        churn_probability=round(float(churn_probability), 4),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
