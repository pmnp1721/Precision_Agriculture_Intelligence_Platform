
# ============================================================
# LangChain @tool wrapper for Tonali's yield model
# ============================================================
import joblib
from langchain.tools import tool

# Load once at agent startup
_BUNDLE = joblib.load("yield_forecast_pipeline.pkl")

@tool
def forecast_crop_yield(
    country: str,
    crop: str,
    year: int,
    rainfall_mm: float,
    temperature_c: float,
    pesticides_tonnes: float,
) -> dict:
    """
    Forecast crop yield in hg/ha for the given country, crop, and year.

    Supports 10 crops: Rice, Maize, Wheat, Potatoes, Cassava, Sorghum,
    Soybeans, Sweet potatoes, Yams, Plantains.

    Returns a dict with predicted yield, confidence metrics, and any warnings.
    """
    from tonali_yield_model import predict_yield  # your inference module
    return predict_yield({
        "country": country,
        "crop": crop,
        "year": year,
        "rainfall_mm": rainfall_mm,
        "temperature_c": temperature_c,
        "pesticides_tonnes": pesticides_tonnes,
    }, bundle=_BUNDLE)
