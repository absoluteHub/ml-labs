from pathlib import Path
from typing import Literal, Optional
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator

MODEL_DIR = Path(__file__).parent / "model"
model = joblib.load(MODEL_DIR / "churn_pipeline.joblib")
model_info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))

app = FastAPI(title="Telco Churn Prediction API", version=model_info["model_version"])

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_services(self):
        services = [self.OnlineSecurity, self.OnlineBackup, self.DeviceProtection,
                    self.TechSupport, self.StreamingTV, self.StreamingMovies]
        if self.InternetService == "No" and any(s != "No internet service" for s in services):
            raise ValueError("Без інтернету всі інтернет-послуги мають бути 'No internet service'")
        if self.InternetService != "No" and any(s == "No internet service" for s in services):
            raise ValueError("Є інтернет, тому значення 'No internet service' неприпустиме")
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError("MultipleLines = 'No phone service' тільки якщо PhoneService = 'No'")
        return self


def postprocess(probability):
    if probability < 0.3:
        risk = "низький"
    elif probability < 0.6:
        risk = "середній"
    else:
        risk = "високий"
    churn_class = int(probability >= 0.5)
    return {
        "churn": "Yes" if churn_class == 1 else "No",
        "churn_class": churn_class,
        "probability": round(float(probability), 4),
        "risk": risk,
        "model_version": model_info["model_version"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": model_info["model_name"], "model_version": model_info["model_version"]}


@app.post("/predict")
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])
    probability = model.predict_proba(X)[0, 1]
    return postprocess(probability)


@app.post("/predict_batch")
def predict_batch(customers: list[Customer]):
    if not 1 <= len(customers) <= 1000:
        raise HTTPException(status_code=422, detail="Кількість клієнтів має бути від 1 до 1000")
    X = pd.DataFrame([c.model_dump() for c in customers])
    probabilities = model.predict_proba(X)[:, 1]
    return {"count": len(customers), "predictions": [postprocess(p) for p in probabilities]}
