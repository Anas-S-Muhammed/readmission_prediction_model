"""
Run this once after training in your notebook to save all model artifacts.
Place this file next to your notebook and run: python save_model.py
"""
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.makedirs("model", exist_ok=True)

# ── Reproduce your notebook pipeline ──────────────────────────────────────────
data = pd.read_csv("TrainingWiDS2021.csv")

threshold = 0.4
data_clean = data.dropna(thresh=len(data) * (1 - threshold), axis=1)

num_cols = data_clean.select_dtypes(include="number").columns
data_clean[num_cols] = data_clean[num_cols].fillna(data_clean[num_cols].median())

data_clean = data_clean.drop([
    "Unnamed: 0", "encounter_id", "hospital_id", "ethnicity", "gender",
    "hospital_admit_source", "icu_admit_source", "icu_id", "icu_type", "pre_icu_los_days"
], axis=1)
data_clean = data_clean.drop(["readmission_status"], axis=1)
data_clean = data_clean.drop(["age_groups", "bmi_groups", "diabetes_rate"], axis=1, errors="ignore")

# Feature engineering — must match main.py exactly
data_clean["diabetes_risk"] = data_clean["age"] * data_clean["bmi"]
data_clean["heart_stress"]  = data_clean["d1_heartrate_max"] - data_clean["d1_heartrate_min"]

# Encode icu_stay_type
le = LabelEncoder()
data_clean["icu_stay_type"] = le.fit_transform(data_clean["icu_stay_type"])

X = data_clean.drop("diabetes_mellitus", axis=1)
y = data_clean["diabetes_mellitus"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)

# ── Save artifacts ─────────────────────────────────────────────────────────────
joblib.dump(rf,                  "model/rf_model.pkl")
joblib.dump(ss,                  "model/scaler.pkl")
joblib.dump(list(X.columns),     "model/feature_columns.pkl")
joblib.dump({"classes": le.classes_.tolist()}, "model/icu_encoder.pkl")

print("✅ Saved 4 files to model/")
print(f"   Features: {X.shape[1]}")
print(f"   ICU types: {le.classes_.tolist()}")
