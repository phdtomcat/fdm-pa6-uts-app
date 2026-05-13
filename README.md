# Dog Bone UTS Predictor

Streamlit app that predicts the Ultimate Tensile Strength (UTS) of 3D-printed PA6
dog-bone specimens from FDM process parameters, and tracks measured values to
validate the underlying ML model in real-world conditions.

## What it does

- **Prediction**: enter infill density, nozzle temperature, and infill pattern -
  get a UTS estimate with an uncertainty band.
- **Validation**: log measured UTS values from actual tensile tests; the app
  computes the deviation from the prediction and stores both into a CSV log.
- **Statistics**: cumulative RMSE / MAE / R², accuracy per engineering tier
  (±0.25, ±0.50, ±1.00, ±2.00 MPa), scatter and error-distribution plots,
  edit and delete operations on individual log entries.

## Model

- **Backend**: LightGBM regressor (Weighted Physics-Augmented Approach i)
- **Features**: infill pattern one-hot, `Aeff` (effective load-bearing area =
  density × geometry-efficiency coefficient), `Dnorm` (Arrhenius diffusion rate
  for PA6, normalized to the training temperature range)
- **Training data**: 33 unique conditions × 3 replicates from a CCD experimental
  design (densities 40–80 %, temps 220–260 °C, three infill geometries)
- **Cross-section**: 4 × 3 mm = 12 mm² gage section, UTS = F_peak / A

If `data/model.txt` is missing, the app falls back to a placeholder physics
formula so the UI still works end-to-end.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`.

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with your
   GitHub account.
3. **New app** → pick this repo, branch, and `app.py` as the entry point.
4. Streamlit Cloud installs `requirements.txt` and starts the app
   (first deploy takes ~2 minutes).
5. Share the resulting `https://<your-app>.streamlit.app` URL.

## Project structure

```
app.py                       Streamlit entry point
modules/
  config.py                  Constants, input ranges, tolerance tiers
  physics.py                 Aeff, Dnorm Arrhenius formulas
  predictor.py               LightGBMUTSModel + PlaceholderUTSModel fallback
  inputs.py                  Shared density/temp/pattern widgets
  validation_log.py          CSV log: append / load / update / delete / clear
  metrics.py                 RMSE / MAE / R² / tier accuracy
  plotting.py                Plotly scatter + error histogram
  sidebar.py                 Model info + quick validation status
  tab_prediction.py          Tab 1
  tab_validation.py          Tab 2
  tab_statistics.py          Tab 3 (with edit/delete UI)
scripts/
  extract_peak_forces.py     Parse raw Time/Force/Stroke CSVs → F_peak
  doe_matrix.py              DOE matrix (Run → density/temp/pattern)
  build_dataset.py           Merge peak forces + DOE → dataset_33.csv
  train_model.py             Train LightGBM, save data/model.txt
data/
  dataset_33.csv             33 conditions × mean UTS + std (training set)
  dataset_99.csv             99 replicates (raw measurements)
  model.txt                  Trained LightGBM (LightGBM native txt format)
  model_features.txt         Ordered feature names used by the model
  validation_log.csv         User-collected measurements (gitignored)
```

## License

MIT
