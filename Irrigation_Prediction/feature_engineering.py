
import numpy as np
import pandas as pd

def add_irrigation_features(df):
    df = df.copy()
    eps = 1e-6

    # -------------------------------------------------
    # 1. Basic water availability features
    # -------------------------------------------------

    df["total_water_input"] = (
        df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    )

    df["rainfall_minus_prev_irr"] = (
        df["Rainfall_mm"] - df["Previous_Irrigation_mm"]
    )

    df["prev_irr_minus_rainfall"] = (
        df["Previous_Irrigation_mm"] - df["Rainfall_mm"]
    )

    df["water_input_per_hectare"] = (
        df["total_water_input"] / (df["Field_Area_hectare"] + eps)
    )

    df["rainfall_per_hectare"] = (
        df["Rainfall_mm"] / (df["Field_Area_hectare"] + eps)
    )

    df["prev_irr_per_hectare"] = (
        df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + eps)
    )

    # -------------------------------------------------
    # 2. Moisture and deficit features
    # -------------------------------------------------

    df["moisture_deficit"] = 100 - df["Soil_Moisture"]

    df["moisture_humidity_ratio"] = (
        df["Soil_Moisture"] / (df["Humidity"] + eps)
    )

    df["rainfall_moisture_ratio"] = (
        df["Rainfall_mm"] / (df["Soil_Moisture"] + eps)
    )

    df["prev_irr_moisture_ratio"] = (
        df["Previous_Irrigation_mm"] / (df["Soil_Moisture"] + eps)
    )

    df["water_input_moisture_ratio"] = (
        df["total_water_input"] / (df["Soil_Moisture"] + eps)
    )

    df["field_moisture_reserve"] = (
        df["Field_Area_hectare"] * df["Soil_Moisture"]
    )

    # -------------------------------------------------
    # 3. Evaporation / water loss proxy features
    # -------------------------------------------------

    df["temp_humidity_index"] = (
        df["Temperature_C"] * df["Humidity"] / 100
    )

    df["temperature_sunlight"] = (
        df["Temperature_C"] * df["Sunlight_Hours"]
    )

    df["sunlight_wind"] = (
        df["Sunlight_Hours"] * df["Wind_Speed_kmh"]
    )

    df["wind_humidity_interaction"] = (
        df["Wind_Speed_kmh"] * df["Humidity"]
    )

    df["evapotranspiration_proxy"] = (
        df["Temperature_C"] 
        * df["Sunlight_Hours"] 
        * (1 + df["Wind_Speed_kmh"] / 20)
        / (df["Humidity"] + eps)
    )

    df["evaporation_pressure"] = (
        df["Temperature_C"] * 0.45
        + df["Sunlight_Hours"] * 0.35
        + df["Wind_Speed_kmh"] * 0.20
    )

    df["dryness_index"] = (
        df["evapotranspiration_proxy"] / (df["total_water_input"] + eps)
    )

    # -------------------------------------------------
    # 4. Rainfall and sunlight relationship
    # -------------------------------------------------

    df["rain_per_sunshine"] = (
        df["Rainfall_mm"] / (df["Sunlight_Hours"] + eps)
    )

    df["temp_rainfall_ratio"] = (
        df["Temperature_C"] / (df["Rainfall_mm"] + eps)
    )

    df["sunlight_rainfall_ratio"] = (
        df["Sunlight_Hours"] / (df["Rainfall_mm"] + eps)
    )

    df["rainfall_humidity_interaction"] = (
        df["Rainfall_mm"] * df["Humidity"]
    )

    # -------------------------------------------------
    # 5. Soil health / salinity interaction features
    # -------------------------------------------------

    df["ec_ph_interaction"] = (
        df["Electrical_Conductivity"] * df["Soil_pH"]
    )

    df["carbon_moisture"] = (
        df["Organic_Carbon"] * df["Soil_Moisture"]
    )

    df["carbon_ph_interaction"] = (
        df["Organic_Carbon"] * df["Soil_pH"]
    )

    df["salinity_moisture_stress"] = (
        df["Electrical_Conductivity"] / (df["Soil_Moisture"] + eps)
    )

    df["soil_health_index"] = (
        df["Organic_Carbon"] * df["Soil_Moisture"]
        / (df["Electrical_Conductivity"] + eps)
    )

    df["ph_deviation_from_neutral"] = (
        abs(df["Soil_pH"] - 7)
    )

    # -------------------------------------------------
    # 6. Net irrigation need proxy
    # -------------------------------------------------

    df["net_water_need"] = (
        df["moisture_deficit"]
        + df["evapotranspiration_proxy"]
        - df["total_water_input"] / 100
    )

    df["water_balance_index"] = (
        df["total_water_input"]
        - df["evapotranspiration_proxy"]
        - df["moisture_deficit"]
    )

    df["water_stress_index"] = (
        df["moisture_deficit"]
        * df["evapotranspiration_proxy"]
        / (df["total_water_input"] + eps)
    )

    # -------------------------------------------------
    # 7. Field scale interaction features
    # -------------------------------------------------

    df["field_rainfall"] = (
        df["Field_Area_hectare"] * df["Rainfall_mm"]
    )

    df["field_previous_irrigation"] = (
        df["Field_Area_hectare"] * df["Previous_Irrigation_mm"]
    )

    df["field_water_input"] = (
        df["Field_Area_hectare"] * df["total_water_input"]
    )

    df["field_evaporation_pressure"] = (
        df["Field_Area_hectare"] * df["evaporation_pressure"]
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df
