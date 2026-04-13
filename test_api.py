#!/usr/bin/env python
"""
Test script to verify the diabetes prediction API works
"""
import requests
import json

API_URL = "http://localhost:8000"

# Sample patient data - using realistic values
test_payload = {
    "age": 58.0,
    "bmi": 28.5,
    "height": 170.0,
    "weight": 82.0,
    "elective_surgery": 0,
    "icu_stay_type": "admit",
    "apache_2_diagnosis": 113.0,
    "apache_3j_diagnosis": 502.0,
    "apache_post_operative": 0,
    "arf_apache": 0,
    "bun_apache": 18.0,
    "creatinine_apache": 0.9,
    "gcs_eyes_apache": 4.0,
    "gcs_motor_apache": 6.0,
    "gcs_unable_apache": 0.0,
    "gcs_verbal_apache": 5.0,
    "glucose_apache": 120.0,
    "heart_rate_apache": 88.0,
    "hematocrit_apache": 38.0,
    "intubated_apache": 0,
    "map_apache": 80.0,
    "resprate_apache": 18.0,
    "sodium_apache": 138.0,
    "temp_apache": 37.0,
    "ventilated_apache": 0,
    "wbc_apache": 9.0,
    "d1_diasbp_max": 85.0,
    "d1_diasbp_min": 55.0,
    "d1_diasbp_noninvasive_max": 85.0,
    "d1_diasbp_noninvasive_min": 55.0,
    "d1_heartrate_max": 105.0,
    "d1_heartrate_min": 62.0,
    "d1_mbp_max": 100.0,
    "d1_mbp_min": 65.0,
    "d1_mbp_noninvasive_max": 100.0,
    "d1_mbp_noninvasive_min": 65.0,
    "d1_resprate_max": 22.0,
    "d1_resprate_min": 12.0,
    "d1_spo2_max": 99.0,
    "d1_spo2_min": 94.0,
    "d1_sysbp_max": 150.0,
    "d1_sysbp_min": 95.0,
    "d1_sysbp_noninvasive_max": 150.0,
    "d1_sysbp_noninvasive_min": 95.0,
    "d1_temp_max": 37.5,
    "d1_temp_min": 36.5,
    "h1_diasbp_max": 82.0,
    "h1_diasbp_min": 58.0,
    "h1_diasbp_noninvasive_max": 82.0,
    "h1_diasbp_noninvasive_min": 58.0,
    "h1_heartrate_max": 100.0,
    "h1_heartrate_min": 68.0,
    "h1_mbp_max": 98.0,
    "h1_mbp_min": 70.0,
    "h1_mbp_noninvasive_max": 98.0,
    "h1_mbp_noninvasive_min": 70.0,
    "h1_resprate_max": 20.0,
    "h1_resprate_min": 14.0,
    "h1_spo2_max": 99.0,
    "h1_spo2_min": 95.0,
    "h1_sysbp_max": 145.0,
    "h1_sysbp_min": 100.0,
    "h1_sysbp_noninvasive_max": 145.0,
    "h1_sysbp_noninvasive_min": 100.0,
    "h1_temp_max": 37.2,
    "h1_temp_min": 36.8,
    "d1_bun_max": 20.0,
    "d1_bun_min": 15.0,
    "d1_calcium_max": 9.5,
    "d1_calcium_min": 8.5,
    "d1_creatinine_max": 1.0,
    "d1_creatinine_min": 0.8,
    "d1_glucose_max": 180.0,
    "d1_glucose_min": 100.0,
    "d1_hco3_max": 26.0,
    "d1_hco3_min": 22.0,
    "d1_hemaglobin_max": 13.5,
    "d1_hemaglobin_min": 11.0,
    "d1_hematocrit_max": 40.0,
    "d1_hematocrit_min": 34.0,
    "d1_platelets_max": 250.0,
    "d1_platelets_min": 180.0,
    "d1_potassium_max": 4.5,
    "d1_potassium_min": 3.8,
    "d1_sodium_max": 140.0,
    "d1_sodium_min": 136.0,
    "d1_wbc_max": 11.0,
    "d1_wbc_min": 7.0,
    "aids": 0,
    "cirrhosis": 0,
    "hepatic_failure": 0,
    "immunosuppression": 0,
    "leukemia": 0,
    "lymphoma": 0,
    "solid_tumor_with_metastasis": 0,
}

def test_api():
    print("=" * 60)
    print("DIABETES PREDICTION API - TEST")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n[1] Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n[2] Testing Root Endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ Root endpoint works")
            print(f"   API: {data['name']}")
            print(f"   Model: {data['model']}")
            print(f"   Features: {data['features']}")
            print(f"   Accuracy: {data['accuracy']}")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 3: Prediction endpoint
    print("\n[3] Testing Prediction Endpoint...")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_payload,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print("✓ Prediction successful!")
            print(f"   Diabetes: {data['diabetes_mellitus']}")
            print(f"   Probability: {data['probability']}")
            print(f"   Risk Level: {data['risk_level']}")
            print(f"   Model: {data['model']}")
            print(f"   Engineered Features:")
            print(f"      - Diabetes Risk: {data['engineered_features']['diabetes_risk']}")
            print(f"      - Heart Stress: {data['engineered_features']['heart_stress']}")
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 4: Features endpoint
    print("\n[4] Testing Features Endpoint...")
    try:
        response = requests.get(f"{API_URL}/features", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ Features endpoint works")
            print(f"   Total Features: {data['total']}")
            print(f"   Engineered Features: {data['engineered']}")
        else:
            print(f"✗ Features endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 5: Invalid icu_stay_type
    print("\n[5] Testing Input Validation...")
    try:
        bad_payload = test_payload.copy()
        bad_payload['icu_stay_type'] = 'invalid_type'
        response = requests.post(
            f"{API_URL}/predict",
            json=bad_payload,
            timeout=10
        )
        if response.status_code == 422:
            print("✓ Validation works (correctly rejected invalid input)")
            print(f"   Error: {response.json()['detail']}")
        else:
            print(f"✗ Should have rejected invalid input but got: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_api()
    exit(0 if success else 1)
