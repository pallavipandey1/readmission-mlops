# FastAPI prediction server — runs inside the Docker container
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Readmission Risk API",
    description="Predicts 30-day hospital readmission risk",
    version="1.0.0"
)

# Load model once at startup — not on every request
MODEL_PATH   = os.getenv("MODEL_PATH",   "model_artifacts/model.pkl")
COLUMNS_PATH = os.getenv("COLUMNS_PATH", "model_artifacts/feature_columns.pkl")

model = None
feature_columns = None

@app.on_event("startup")
def load_model():
    global model, feature_columns
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    print(f"Model loaded. Features: {feature_columns}")

# Define what the input data looks like
class PatientData(BaseModel):
    age_num:              float = 65.0
    time_in_hospital:     float = 5.0
    num_lab_procedures:   float = 40.0
    num_procedures:       float = 1.0
    num_medications:      float = 15.0
    number_diagnoses:     float = 7.0
    number_outpatient:    float = 0.0
    number_emergency:     float = 1.0
    number_inpatient:     float = 2.0
    total_visits:         float = 3.0
    meds_changed:         float = 1.0
    meds_on:              float = 5.0
    a1c_tested:           float = 1.0
    a1c_high:             float = 1.0
    discharged_home:      float = 1.0
    admitted_emergency:   float = 0.0
    has_circulatory:      float = 0.0
    has_diabetes_diag:    float = 1.0
    diag_1:               float = 250.0
    diag_2:               float = 401.0
    diag_3:               float = 0.0

@app.get("/health")
def health():
    """Health check — used by Docker and load balancers"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(patient: PatientData):
    """Return readmission risk score for one patient"""
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")
    row = pd.DataFrame([patient.dict()])
    row = row[feature_columns]  # ensure correct column order
    prob = model.predict_proba(row)[0][1]
    risk = ("HIGH" if prob >= 0.5 else
            "MEDIUM" if prob >= 0.3 else "LOW")
    return {
        "readmission_probability": round(float(prob), 4),
        "risk_level":              risk,
        "threshold_used":          0.5
    }