import warnings

import joblib
import numpy as np
import pandas as pd

from .paths import YIELD_DIR


BUNDLE_PATH = YIELD_DIR / "deploy" / "yield_forecast_pipeline.pkl"
DATA_PATH = YIELD_DIR / "yield_df_cleaned.csv"

CONFIDENCE_TIERS = {
    "high": 1.00,
    "medium": 0.90,
    "low": 0.75,
    "very_low": 0.55,
}


def load_yield_bundle():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(BUNDLE_PATH)


def load_yield_dataset():
    return pd.read_csv(DATA_PATH)


def get_supported_crops():
    return list(load_yield_bundle()["known_crops"])


def get_supported_countries():
    return list(load_yield_bundle()["known_countries"])


def forecast_crop_yield(payload):
    bundle = load_yield_bundle()
    warnings_list = []
    meta = bundle["metadata"]

    country = str(payload["country"]).strip()
    crop = str(payload["crop"]).strip()
    year = int(payload["year"])
    rain = float(payload["rainfall_mm"])
    temp = float(payload["temperature_c"])
    pest = float(payload["pesticides_tonnes"])

    flag_unknown_country = country not in bundle["known_countries"]
    flag_extrapolated_year = year > meta["year_max_trained"] + 5
    flag_stale_lag = False
    flag_no_history = False

    if flag_unknown_country:
        warnings_list.append(
            f"Country '{country}' was not in training data; using global patterns."
        )
    if crop not in bundle["known_crops"]:
        raise ValueError(f"Crop '{crop}' is not supported by the yield model.")
    if flag_extrapolated_year:
        warnings_list.append(
            f"Year {year} is beyond the training range ending in {meta['year_max_trained']}."
        )
    if rain < 0 or temp < -30 or temp > 50 or pest < 0:
        warnings_list.append("One or more values are outside a plausible agronomic range.")

    key = (country, crop)
    lookup_row = bundle["latest_lookup"].get(key)

    if payload.get("yield_lag_1") is not None:
        lag_1 = float(payload["yield_lag_1"])
    elif lookup_row is not None:
        lag_1 = float(lookup_row["latest_yield"])
        lag_year = int(lookup_row["latest_year"])
        lag_age = year - lag_year
        warnings_list.append(
            f"Using latest observed yield {int(lag_1):,} hg/ha from {lag_year} as lag input."
        )
        if lag_age > 3:
            flag_stale_lag = True
    else:
        flag_no_history = True
        lag_1 = float(meta["y_median"])
        warnings_list.append(
            f"No prior history for ({country}, {crop}); using global median yield."
        )

    if payload.get("rainfall_roll3") is not None:
        rain_roll3 = float(payload["rainfall_roll3"])
    elif lookup_row is not None:
        rain_roll3 = float(lookup_row["latest_roll3"])
    else:
        rain_roll3 = rain

    opt_rain = bundle["crop_optimal_rainfall_mm"].get(crop, 1000)
    opt_temp = bundle["crop_optimal_temp_c"].get(crop, 22.0)

    row = pd.DataFrame(
        [
            {
                "Year": year,
                "average_rain_fall_mm_per_year": rain,
                "pesticides_tonnes": pest,
                "avg_temp": temp,
                "rainfall_adequacy_index": rain / opt_rain,
                "thermal_stress_score": abs(temp - opt_temp),
                "log_pesticides": float(np.log1p(pest)),
                "yield_lag_1": lag_1,
                "rainfall_roll3": rain_roll3,
                "year_index": year - meta["year_anchor"],
                "Area": country,
                "Item": crop,
            }
        ]
    )[bundle["numeric_features"] + bundle["categorical_features"]]

    pred = float(bundle["pipeline"].predict(row)[0])
    upper_cap = meta["y_max"] * 1.5

    if pred < 0:
        warnings_list.append(f"Raw prediction {pred:.0f} was negative and clipped to 0.")
        pred = 0.0
    elif pred > upper_cap:
        warnings_list.append("Raw prediction exceeded the training cap and was clipped.")
        pred = upper_cap

    if flag_no_history or flag_unknown_country:
        tier = "very_low"
    elif flag_stale_lag and flag_extrapolated_year:
        tier = "low"
    elif flag_stale_lag or flag_extrapolated_year:
        tier = "medium"
    else:
        tier = "high"

    return {
        "predicted_yield_hg_per_ha": int(round(pred)),
        "confidence_r2": round(meta["confidence_r2_baseline"] * CONFIDENCE_TIERS[tier], 4),
        "confidence_tier": tier,
        "confidence_rmse_hg_ha": meta["confidence_rmse_hg_ha"],
        "confidence_mape_%": meta["confidence_mape_%"],
        "unit": meta["unit"],
        "model_version": meta["model_name"],
        "warnings": warnings_list,
        "features": row.iloc[0].to_dict(),
    }
