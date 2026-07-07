import pickle
import warnings

import pandas as pd

from .paths import CROP_DIR


CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
MODEL_PATH = CROP_DIR / "crop_recommendation_model.pkl"
DATA_PATH = CROP_DIR / "Crop_recommendation.csv"


def load_crop_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)


def load_crop_dataset():
    return pd.read_csv(DATA_PATH)


def recommend_crop(payload):
    model = load_crop_model()
    row = pd.DataFrame([{feature: float(payload[feature]) for feature in CROP_FEATURES}])
    prediction = model.predict(row)[0]

    probabilities = []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        probabilities = sorted(
            [
                {"crop": str(crop), "probability": float(score)}
                for crop, score in zip(model.classes_, proba)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )

    return {
        "recommended_crop": str(prediction),
        "top_predictions": probabilities[:5],
        "input": row.iloc[0].to_dict(),
    }
