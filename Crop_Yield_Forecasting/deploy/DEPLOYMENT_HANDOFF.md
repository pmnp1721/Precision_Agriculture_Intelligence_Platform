
# Yield Forecasting Model — Deployment Handoff
Prepared by: Tonali Gupta
Model: lasso_v1 (Lasso Regression (L1))
Trained: 2026-07-07T05:48:14.648646+00:00
Training window: 1991–2013 (12,532 rows)

## Artifact
Path: .\deploy\yield_forecast_pipeline.pkl
Size: 29.0 KB
Contains: sklearn Pipeline (StandardScaler + OneHotEncoder + Lasso), agronomic constant
lookups, latest-yield lookup per (country, crop), scope lists, metadata.

## Load
    import joblib
    bundle = joblib.load("yield_forecast_pipeline.pkl")

## Call (JSON in → JSON out)
Input:
    {
      "country"           : "India",
      "crop"              : "Rice",
      "year"              : 2026,
      "rainfall_mm"       : 1200,
      "temperature_c"     : 27.5,
      "pesticides_tonnes" : 45.2
    }

Output (for a fresh, in-scope query — confidence_tier='high'):
    {
      "predicted_yield_hg_per_ha": 42150,
      "confidence_r2"            : 0.9627,
      "confidence_tier"          : "high",
      "confidence_rmse_hg_ha"    : 17044,
      "confidence_mape_%"        : 15.51,
      "unit"                     : "hg/ha",
      "model_version"            : "lasso_v1",
      "warnings"                 : []
    }

Output (for a typical agent query — 13-yr-stale lag + 2026 extrapolation):
    {
      "predicted_yield_hg_per_ha": 38307,
      "confidence_r2"            : 0.7220,
      "confidence_tier"          : "low",
      "confidence_rmse_hg_ha"    : 17044,
      "confidence_mape_%"        : 15.51,
      "unit"                     : "hg/ha",
      "model_version"            : "lasso_v1",
      "warnings"                 : ["Year 2026 is >5 years beyond training range...", "yield_lag_1 auto-filled from 2013..."]
    }

## Supported Scope
- Crops (10)      : Rice, Maize, Wheat, Potatoes, Cassava, Sorghum, Soybeans,
                    Sweet potatoes, Yams, Plantains
- Countries       : 101 known (see bundle['known_countries'])
- Training years  : 1991–2013
- Inference years : any year; warnings issued for >5-yr extrapolation

## Guardrails Built In
- Unknown country      → warning + zero-vector fallback + confidence_tier='very_low'
- Unknown crop         → hard error (must be one of the 10 supported crops)
- Missing `yield_lag_1`→ auto-filled from latest observed yield for that (country, crop)
- Stale lag (>3 yr old)→ warning + confidence_tier degraded ('medium' or 'low')
- Year > train_max + 5 → warning + confidence_tier degraded ('medium' or 'low')
- No prior history     → global-median fallback + confidence_tier='very_low'
- Negative prediction  → clipped to 0 with warning
- Absurdly-high pred   → clipped to 1.5× max observed yield with warning
- Bad inputs (negative rain, extreme temps) → warning

## Confidence Interpretation
- Baseline R²      = 0.9627 (test set, 2011–2013, with 1-year lag)
- Baseline RMSE    = 17,044 hg/ha (~± this many hg/ha typical error)
- Baseline MAPE    = 15.51% (typical proportional error)
- **Confidence tiers** (degradation applied automatically):
  - `high`     = 1.00 × baseline R²  →  0.9627
  - `medium`   = 0.90 × baseline R²  →  0.8664  (stale lag OR extrapolated year)
  - `low`      = 0.75 × baseline R²  →  0.7220  (stale lag AND extrapolated year)
  - `very_low` = 0.55 × baseline R²  →  0.5295  (unknown country OR no lag history)
- Consumers should check `confidence_tier` and `warnings[]` fields — do not rely on `confidence_r2` alone.

## Known Limitations
- Model was trained 1990–2013; predictions for 2020+ are temporal extrapolation.
- Root crops (Potatoes, Cassava, Sweet potatoes) have wider absolute error but
  low percentage error. Cereals (Sorghum, Soybeans) have larger MAPE.
- Cold-start countries (never seen in training) get global-pattern fallback —
  Sudan MAPE ~10% in evaluation. Not recommended for zero-history countries.
- Model assumes national-scale inputs; sub-national predictions may drift.

## Retraining / Update Path
Rerun `yield_forecasting_tonali.ipynb` Sections 3 → 6 after appending new FAO/WDB
data to `yield_df.csv`. No code changes needed as long as the schema matches.
