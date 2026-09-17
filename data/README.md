# Data

This repository does **not** include the raw meteorological data. Three of the four stations used in the study (Sinop, İnceburun, Airport) are official Turkish State Meteorological Service (MGM) stations; that data is state-owned and is not redistributed here. The fourth station ("OB") is a personal weather station, kept out of the repo for consistency.

What matters for evaluating this project is not the raw dataset itself but the **method**: how the data is quality-controlled, how ramp-up events are detected, how the feature matrix is built, and how the models are compared — all of which is in `src/`, fully runnable once you point it at data in the schema below.

## Expected schema

To run the pipeline on your own station data, place CSV files here matching this shape:

**Station files** (one per station, e.g. `sinop.csv`, `inceburun.csv`, `airport.csv`, `ob.csv`), each with a `datetime` column plus per-station columns:

| column | meaning |
|---|---|
| `datetime` | timestamp, parseable by `pandas.to_datetime` |
| `..._basinc_hpa` | station pressure (hPa) |
| `..._deniz_basinc_hpa` | sea-level pressure (hPa) |
| `..._sicaklik_c` | temperature (°C) |
| `..._yagis_mm` | precipitation (mm) |
| `..._wind_speed_ms` | wind speed (m/s) |
| `..._wind_dir_deg` | wind direction (degrees) |

**Cleaned output** (`wl_temizlenmis.csv`, produced by `src/quality_control/`): a single continuous, minute-resolution series with a `ws_clean_ms` column (quality-controlled wind speed) that the rest of the pipeline (ramp detection, std analysis, feature matrix) reads from.

## Pipeline order

1. `src/quality_control/` — raw station data in, quality-controlled minute series out
2. `src/ramp_detection/` — ramp-up event detection, run on the **raw** (not yet cleaned) series — see the paper for why
3. `src/feature_matrix/` — builds the 24h × 4 stations × 6 variables input matrix per detected event
4. `src/std_analysis/` — builds the natural-variability reference table used later for HIT/MISS evaluation
5. `src/models/` — trains and compares 6 models (CNN-LSTM, LSTM, GRU, XGBoost, CatBoost, LightGBM) on the feature matrix
