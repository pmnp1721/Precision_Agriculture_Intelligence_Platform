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

    return pd.DataFrame(
        {
            "Irrigation_Need": ["Low", "Medium", "High"],
            "Region": ["Unknown", "Unknown", "Unknown"],
        }
    )


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
