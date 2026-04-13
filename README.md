# Diabetes Mellitus Prediction Model

A machine learning service that predicts whether an ICU patient has diabetes mellitus using the WiDS Datathon 2021 dataset. The model is deployed as a FastAPI service with a trained Random Forest classifier.

## Overview

This project includes:
- A Jupyter notebook for model training and experimentation
- A FastAPI service for making predictions
- Model artifacts (Random Forest, scaler, feature encoders) saved for production use

The model achieves **82.2% accuracy** and an **F1 score of 0.46** on the test set.

## Dataset

- **Source:** WiDS Datathon 2021 (Kaggle)
- **Patients:** 130,157 ICU admissions
- **Features:** 181 raw features (95 after cleaning)
- **Target:** Diabetes mellitus (binary classification)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train and save the model:
   ```bash
   python save_model.py
   ```
   This creates the model artifacts in the `model/` directory.

4. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```
   The server runs on `http://localhost:8000`

## API Usage

Access the interactive API documentation at `http://localhost:8000/docs` (Swagger UI).

### Endpoints

- `GET /` — Service info and model metrics
- `GET /health` — Health check
- `POST /predict` — Predict diabetes mellitus for a patient
- `GET /features` — List all model features

### Example Request

```python
import requests

payload = {
    "age": 58.0,
    "bmi": 28.5,
    "height": 170.0,
    "weight": 82.0,
    "elective_surgery": 0,
    "icu_stay_type": "admit",
    # ... include all 95 features
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

### Example Response

```json
{
    "diabetes_mellitus": true,
    "probability": 0.72,
    "risk_level": "High",
    "engineered_features": {
        "diabetes_risk": 1624.0,
        "heart_stress": 43.0
    },
    "model": "Random Forest (WiDS 2021)"
}
```
## Model Performance

| Metric | Score |
|--------|-------|
| Random Forest Accuracy | 82.2% |
| Random Forest F1 Score | 0.46 |
| Best Model | Random Forest |

The Random Forest outperformed both Logistic Regression (81% acc, F1: 0.39) and Decision Trees (75% acc, F1: 0.43).

## Data Preprocessing

- Removed columns with >40% missing values
- Filled numeric columns with median values
- Dropped non-predictive columns (encounter ID, hospital ID, demographic info)
- Encoded categorical variable (`icu_stay_type`)
- Applied standard scaling before model training

## Feature Engineering

Two engineered features improve predictions:
- **diabetes_risk:** age × BMI
- **heart_stress:** d1_heartrate_max − d1_heartrate_min

## Known Issues & Fixes

### Pydantic v2 Compatibility
The `main.py` API was updated to use Pydantic v2's `.model_dump()` method instead of the deprecated Pydantic v1 `.dict()` method. This ensures compatibility with `pydantic==2.8.2`.

## Challenges

- **Imbalanced data:** ~25% diabetic vs ~75% non-diabetic patients
- **Feature volume:** 181 raw features required aggressive filtering
- **Recall tradeoff:** Model prioritizes precision over detecting all diabetic patients

## Project Structure

```
.
├── main.py                          # FastAPI service
├── save_model.py                    # Model training & artifact saving
├── peadmission_predictor.ipynb      # Jupyter notebook (exploration & training)
├── requirements.txt                 # Python dependencies
├── TrainingWiDS2021.csv             # Dataset
├── model/
│   ├── rf_model.pkl                 # Trained Random Forest
│   ├── scaler.pkl                   # StandardScaler for normalization
│   ├── feature_columns.pkl          # Feature order
│   └── icu_encoder.pkl              # ICU type encoder
└── README.md
```

## Next Steps

- Implement SMOTE for class imbalance handling
- Experiment with XGBoost and LightGBM models
- Add request validation and error handling
- Deploy to cloud (AWS, Azure, or GCP)
- Monitor model drift in production
