# colombo-rain-study-ML
Assignment targeting the applied machine learning concepts attended during the MSc program at the University of Moratuwa.

Next-day rain classification for the Colombo area using regularised Quadratic Discriminant Analysis (QDA) and SHAP. Pansilu Wijesiri, 269220E.

## Run the application

Use Python 3.12. The saved model and dataset are included.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Reproduce the results

```bash
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.explain
```

## Dataset

The included snapshot contains 4,383 daily ERA5 records for 2014–2025, obtained through the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). The requested coordinates were 6.9271 N, 79.8612 E; the returned grid cell was 7.0 N, 80.0 E. These are gridded estimates for the Colombo area, not station measurements. Source details, units and the request are in `data/provenance.json`.

Weather data by Open-Meteo, using ECMWF ERA5, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Daily data were converted to CSV and weather-history and seasonal features were derived. No personal data were collected.

## Deployment

In Streamlit Community Cloud, select `PansiluAW/colombo-rain-study-ML`, branch `main`, entry point `streamlit_app.py` and Python 3.12. No API key or secrets are needed. The application demonstrates historical classification using delayed reanalysis; it is not a live forecast or flood warning.
