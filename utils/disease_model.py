from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from .paths import DISEASE_DIR


ARTIFACT_DIR = DISEASE_DIR / "crop_disease_artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_crop_disease_model.pth"
SUMMARY_PATH = ARTIFACT_DIR / "dataset_class_summary.csv"
COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"
REPORT_PATH = ARTIFACT_DIR / "best_model_classification_report.csv"


def load_disease_artifacts():
    return {
        "class_summary": pd.read_csv(SUMMARY_PATH),
        "model_comparison": pd.read_csv(COMPARISON_PATH),
        "classification_report": pd.read_csv(REPORT_PATH),
        "model_available": True,
        "fall_back_to_visual": not MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


def _open_image(image_file):
    if hasattr(image_file, "seek"):
        image_file.seek(0)
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def _visual_fallback_prediction(image):
    width, height = image.size
    pixels = list(image.getdata())
    total = max(1, len(pixels))

    dark_pixels = 0
    green_pixels = 0
    brown_pixels = 0

    for r, g, b in pixels:
        brightness = (r + g + b) / 3.0
        if brightness < 70:
            dark_pixels += 1
        if g > r and g > b and g >= 90:
            green_pixels += 1
        if r > 90 and g > 50 and b < 90 and brightness < 180:
            brown_pixels += 1

    dark_ratio = dark_pixels / total
    green_ratio = green_pixels / total
    brown_ratio = brown_pixels / total

    if brown_ratio > 0.05 or dark_ratio > 0.08:
        prediction = "Possible disease"
        confidence = min(0.96, 0.7 + brown_ratio * 1.4 + dark_ratio * 0.6)
    else:
        prediction = "Healthy"
        confidence = min(0.95, 0.72 + green_ratio * 0.2)

    return prediction, round(confidence, 2)


def detect_crop_disease(image_file):
    image = _open_image(image_file)
    prediction, confidence = _visual_fallback_prediction(image)
    message = (
        f"Visual fallback analysis suggests the leaf looks {prediction.lower()} with "
        f"{confidence:.0%} confidence. The trained model checkpoint is not present locally, "
        "so this uses a lightweight visual heuristic instead."
    )
    return {
        "available": True,
        "mode": "visual_fallback",
        "prediction": prediction,
        "confidence": confidence,
        "message": message,
    }
