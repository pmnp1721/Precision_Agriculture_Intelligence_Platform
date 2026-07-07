import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

from .paths import IRRIGATION_DIR


MODEL_PATH = IRRIGATION_DIR / "best_xgboost_irrigation_pipeline.pkl"
ENCODER_PATH = IRRIGATION_DIR / "target_encoder.pkl"
DATA_PATHS = [
    IRRIGATION_DIR / "IrrigationNeedDataset" / "train.csv",
    Path("/mount/src/precision_agriculture_intelligence_platform") / "Irrigation_Prediction" / "IrrigationNeedDataset" / "train.csv",
    Path("/app") / "Irrigation_Prediction" / "IrrigationNeedDataset" / "train.csv",
]

BASE_FEATURES = [
    "Soil_Type",
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Field_Area_hectare",
    "Mulching_Used",
    "Previous_Irrigation_mm",
    "Region",
]


def _import_feature_engineering():
    path = str(IRRIGATION_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    from feature_engineering import add_irrigation_features

    return add_irrigation_features


def load_irrigation_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)


def load_target_encoder():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(ENCODER_PATH)


def load_irrigation_dataset(sample_rows=None):
    for data_path in DATA_PATHS:
        if data_path.exists():
            if sample_rows:
                return pd.read_csv(data_path, nrows=sample_rows)
            return pd.read_csv(data_path)

    fallback_rows = [
        {
            "Soil_Type": "Loamy",
            "Soil_pH": 6.1,
            "Soil_Moisture": 34.0,
            "Organic_Carbon": 1.02,
            "Electrical_Conductivity": 2.5,
            "Temperature_C": 24.8,
            "Humidity": 61.0,
            "Rainfall_mm": 820.0,
            "Sunlight_Hours": 6.8,
            "Wind_Speed_kmh": 13.2,
            "Crop_Type": "Wheat",
            "Crop_Growth_Stage": "Vegetative",
            "Season": "Kharif",
            "Irrigation_Type": "Drip",
            "Water_Source": "Rainwater",
            "Field_Area_hectare": 3.2,
            "Mulching_Used": "Yes",
            "Previous_Irrigation_mm": 42.0,
            "Region": "North",
            "Irrigation_Need": "Low",
        },
        {
            "Soil_Type": "Clay",
            "Soil_pH": 7.0,
            "Soil_Moisture": 48.0,
            "Organic_Carbon": 0.68,
            "Electrical_Conductivity": 1.8,
            "Temperature_C": 28.5,
            "Humidity": 74.0,
            "Rainfall_mm": 640.0,
            "Sunlight_Hours": 7.3,
            "Wind_Speed_kmh": 9.8,
            "Crop_Type": "Rice",
            "Crop_Growth_Stage": "Sowing",
            "Season": "Rabi",
            "Irrigation_Type": "Sprinkler",
            "Water_Source": "River",
            "Field_Area_hectare": 5.5,
            "Mulching_Used": "No",
            "Previous_Irrigation_mm": 38.0,
            "Region": "South",
            "Irrigation_Need": "Medium",
        },
        {
            "Soil_Type": "Sandy",
            "Soil_pH": 5.6,
            "Soil_Moisture": 19.0,
            "Organic_Carbon": 0.48,
            "Electrical_Conductivity": 0.95,
            "Temperature_C": 33.2,
            "Humidity": 46.0,
            "Rainfall_mm": 310.0,
            "Sunlight_Hours": 8.9,
            "Wind_Speed_kmh": 18.0,
            "Crop_Type": "Maize",
            "Crop_Growth_Stage": "Flowering",
            "Season": "Zaid",
            "Irrigation_Type": "Flood",
            "Water_Source": "Canal",
            "Field_Area_hectare": 2.8,
            "Mulching_Used": "No",
            "Previous_Irrigation_mm": 55.0,
            "Region": "West",
            "Irrigation_Need": "High",
        },
    ]
    fallback_df = pd.DataFrame(fallback_rows)
    if sample_rows:
        return fallback_df.head(sample_rows)
    return fallback_df


def predict_irrigation(payload):
    model = load_irrigation_model()
    encoder = load_target_encoder()
    add_irrigation_features = _import_feature_engineering()

    row = pd.DataFrame([{feature: payload[feature] for feature in BASE_FEATURES}])
    enriched = add_irrigation_features(row)

    raw_prediction = model.predict(enriched)[0]
    label = str(encoder.inverse_transform([raw_prediction])[0])

    probabilities = []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(enriched)[0]
        probabilities = [
            {"need": str(label), "probability": float(score)}
            for label, score in zip(encoder.classes_, proba)
        ]
        probabilities = sorted(probabilities, key=lambda item: item["probability"], reverse=True)

    stress = enriched.iloc[0][
        [
            "moisture_deficit",
            "evapotranspiration_proxy",
            "dryness_index",
            "water_balance_index",
            "water_stress_index",
        ]
    ].to_dict()

    return {
        "irrigation_need": label,
        "probabilities": probabilities,
        "stress_indicators": {key: float(value) for key, value in stress.items()},
        "input": row.iloc[0].to_dict(),
    }
