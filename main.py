from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import numpy as np
import joblib
import os

# ── Load model artifacts once at startup ──────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")

try:
    rf_model        = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    scaler          = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    icu_encoder     = joblib.load(os.path.join(MODEL_DIR, "icu_encoder.pkl"))
    ICU_CLASSES     = icu_encoder["classes"]   # ['admit', 'readmit', 'transfer']
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model file not found: {e}\n"
        "Run the notebook first and make sure model/*.pkl files exist."
    )

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🏥 Diabetes Predictor API",
    description=(
        "Predicts whether an ICU patient has diabetes mellitus "
        "using a Random Forest trained on the WiDS Datathon 2021 dataset."
    ),
    version="1.0.0",
)


# ── Input schema ──────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    """All 95 raw features the model was trained on.
    diabetes_risk and heart_stress are computed automatically — do NOT include them."""

    # Demographics & admission
    age:                         float = Field(..., example=58.0,  description="Patient age in years")
    bmi:                         float = Field(..., example=28.5,  description="Body mass index")
    height:                      float = Field(..., example=170.0, description="Height in cm")
    weight:                      float = Field(..., example=82.0,  description="Weight in kg")
    elective_surgery:            int   = Field(..., example=0,     description="1 = elective surgery admission")
    icu_stay_type:               str   = Field(..., example="admit", description="One of: admit, readmit, transfer")

    # APACHE scores
    apache_2_diagnosis:          float = Field(..., example=113.0)
    apache_3j_diagnosis:         float = Field(..., example=502.0)
    apache_post_operative:       int   = Field(..., example=0)
    arf_apache:                  int   = Field(..., example=0)
    bun_apache:                  float = Field(..., example=18.0)
    creatinine_apache:           float = Field(..., example=0.9)
    gcs_eyes_apache:             float = Field(..., example=4.0)
    gcs_motor_apache:            float = Field(..., example=6.0)
    gcs_unable_apache:           float = Field(..., example=0.0)
    gcs_verbal_apache:           float = Field(..., example=5.0)
    glucose_apache:              float = Field(..., example=120.0)
    heart_rate_apache:           float = Field(..., example=88.0)
    hematocrit_apache:           float = Field(..., example=38.0)
    intubated_apache:            int   = Field(..., example=0)
    map_apache:                  float = Field(..., example=80.0)
    resprate_apache:             float = Field(..., example=18.0)
    sodium_apache:               float = Field(..., example=138.0)
    temp_apache:                 float = Field(..., example=37.0)
    ventilated_apache:           int   = Field(..., example=0)
    wbc_apache:                  float = Field(..., example=9.0)

    # Day 1 vitals — max/min
    d1_diasbp_max:               float = Field(..., example=85.0)
    d1_diasbp_min:               float = Field(..., example=55.0)
    d1_diasbp_noninvasive_max:   float = Field(..., example=85.0)
    d1_diasbp_noninvasive_min:   float = Field(..., example=55.0)
    d1_heartrate_max:            float = Field(..., example=105.0, description="Used to compute heart_stress")
    d1_heartrate_min:            float = Field(..., example=62.0,  description="Used to compute heart_stress")
    d1_mbp_max:                  float = Field(..., example=100.0)
    d1_mbp_min:                  float = Field(..., example=65.0)
    d1_mbp_noninvasive_max:      float = Field(..., example=100.0)
    d1_mbp_noninvasive_min:      float = Field(..., example=65.0)
    d1_resprate_max:             float = Field(..., example=22.0)
    d1_resprate_min:             float = Field(..., example=12.0)
    d1_spo2_max:                 float = Field(..., example=99.0)
    d1_spo2_min:                 float = Field(..., example=94.0)
    d1_sysbp_max:                float = Field(..., example=150.0)
    d1_sysbp_min:                float = Field(..., example=95.0)
    d1_sysbp_noninvasive_max:    float = Field(..., example=150.0)
    d1_sysbp_noninvasive_min:    float = Field(..., example=95.0)
    d1_temp_max:                 float = Field(..., example=37.5)
    d1_temp_min:                 float = Field(..., example=36.5)

    # Hour 1 vitals — max/min
    h1_diasbp_max:               float = Field(..., example=82.0)
    h1_diasbp_min:               float = Field(..., example=58.0)
    h1_diasbp_noninvasive_max:   float = Field(..., example=82.0)
    h1_diasbp_noninvasive_min:   float = Field(..., example=58.0)
    h1_heartrate_max:            float = Field(..., example=100.0)
    h1_heartrate_min:            float = Field(..., example=68.0)
    h1_mbp_max:                  float = Field(..., example=98.0)
    h1_mbp_min:                  float = Field(..., example=70.0)
    h1_mbp_noninvasive_max:      float = Field(..., example=98.0)
    h1_mbp_noninvasive_min:      float = Field(..., example=70.0)
    h1_resprate_max:             float = Field(..., example=20.0)
    h1_resprate_min:             float = Field(..., example=14.0)
    h1_spo2_max:                 float = Field(..., example=99.0)
    h1_spo2_min:                 float = Field(..., example=95.0)
    h1_sysbp_max:                float = Field(..., example=145.0)
    h1_sysbp_min:                float = Field(..., example=100.0)
    h1_sysbp_noninvasive_max:    float = Field(..., example=145.0)
    h1_sysbp_noninvasive_min:    float = Field(..., example=100.0)
    h1_temp_max:                 float = Field(..., example=37.2)
    h1_temp_min:                 float = Field(..., example=36.8)

    # Day 1 lab values — max/min
    d1_bun_max:                  float = Field(..., example=20.0)
    d1_bun_min:                  float = Field(..., example=15.0)
    d1_calcium_max:              float = Field(..., example=9.5)
    d1_calcium_min:              float = Field(..., example=8.5)
    d1_creatinine_max:           float = Field(..., example=1.0)
    d1_creatinine_min:           float = Field(..., example=0.8)
    d1_glucose_max:              float = Field(..., example=180.0)
    d1_glucose_min:              float = Field(..., example=100.0)
    d1_hco3_max:                 float = Field(..., example=26.0)
    d1_hco3_min:                 float = Field(..., example=22.0)
    d1_hemaglobin_max:           float = Field(..., example=13.5)
    d1_hemaglobin_min:           float = Field(..., example=11.0)
    d1_hematocrit_max:           float = Field(..., example=40.0)
    d1_hematocrit_min:           float = Field(..., example=34.0)
    d1_platelets_max:            float = Field(..., example=250.0)
    d1_platelets_min:            float = Field(..., example=180.0)
    d1_potassium_max:            float = Field(..., example=4.5)
    d1_potassium_min:            float = Field(..., example=3.8)
    d1_sodium_max:               float = Field(..., example=140.0)
    d1_sodium_min:               float = Field(..., example=136.0)
    d1_wbc_max:                  float = Field(..., example=11.0)
    d1_wbc_min:                  float = Field(..., example=7.0)

    # Comorbidities (0 or 1)
    aids:                        int   = Field(..., example=0)
    cirrhosis:                   int   = Field(..., example=0)
    hepatic_failure:             int   = Field(..., example=0)
    immunosuppression:           int   = Field(..., example=0)
    leukemia:                    int   = Field(..., example=0)
    lymphoma:                    int   = Field(..., example=0)
    solid_tumor_with_metastasis: int   = Field(..., example=0)


# ── Output schema ─────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    diabetes_mellitus: bool
    probability:       float
    risk_level:        str
    engineered_features: dict
    model:             str


# ── Helper: encode + build feature vector ─────────────────────────────────────
def build_feature_vector(patient: PatientData) -> np.ndarray:
    """Apply the same feature engineering as the notebook, then align to
    the exact column order the scaler/model expects."""

    # Validate icu_stay_type
    if patient.icu_stay_type not in ICU_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"icu_stay_type must be one of {ICU_CLASSES}, got '{patient.icu_stay_type}'"
        )

    data = patient.model_dump()

    # 1. Encode categorical — same as LabelEncoder used in training
    data["icu_stay_type"] = ICU_CLASSES.index(patient.icu_stay_type)

    # 2. Feature engineering — MUST match the notebook exactly
    data["diabetes_risk"] = data["age"] * data["bmi"]
    data["heart_stress"]  = data["d1_heartrate_max"] - data["d1_heartrate_min"]

    # 3. Align to training column order
    vector = [data[col] for col in feature_columns]
    return np.array(vector).reshape(1, -1)


def probability_to_risk(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.55:
        return "Moderate"
    else:
        return "High"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":        "Diabetes Predictor API",
        "model":       "Random Forest — WiDS Datathon 2021",
        "features":    len(feature_columns),
        "accuracy":    "82.2%",
        "f1_score":    "0.46",
        "docs":        "/docs",
        "health":      "/health",
    }


@app.get("/health")
def health():
    return {
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": rf_model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    """
    Predict whether an ICU patient has diabetes mellitus.

    - Returns `diabetes_mellitus`: true / false
    - Returns `probability`: confidence (0.0 – 1.0)
    - Returns `risk_level`: Low / Moderate / High
    """
    X = build_feature_vector(patient)
    X_scaled = scaler.transform(X)

    prediction   = bool(rf_model.predict(X_scaled)[0])
    probability  = round(float(rf_model.predict_proba(X_scaled)[0][1]), 4)
    risk_level   = probability_to_risk(probability)

    return PredictionResponse(
        diabetes_mellitus=prediction,
        probability=probability,
        risk_level=risk_level,
        engineered_features={
            "diabetes_risk": round(patient.age * patient.bmi, 2),
            "heart_stress":  round(patient.d1_heartrate_max - patient.d1_heartrate_min, 2),
        },
        model="Random Forest (WiDS 2021)",
    )


@app.get("/features")
def features():
    """Returns the list of all features the model uses, in order."""
    return {
        "total":    len(feature_columns),
        "columns":  feature_columns,
        "engineered": ["diabetes_risk (age × bmi)", "heart_stress (d1_heartrate_max − min)"],
    }