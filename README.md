# colombo-rain-study-ML
Next-day rain classification for the Colombo area using regularised Quadratic Discriminant Analysis (QDA) and exact interventional SHAP.

IT5514 - Applied Machine Learning. Pansilu Ashinshana Wijesiri, 269220E. MSc in Artificial Intelligence, University of Moratuwa.

[Live application on Azure](https://colombo-rain-269220e-h4exayebdwandnc2.centralindia-01.azurewebsites.net/) | [Project links and screenshots](PROJECT_LINKS.md)

## Run the application

Use Python 3.12. The saved model and dataset are included.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m flask --app app run
```

Open `http://127.0.0.1:5000`. On Windows, activate the environment with `.venv\Scripts\activate`.

## Reproduce the results

```bash
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.explain
python -m unittest discover -s tests -v
```

## Dataset

The included snapshot contains 4,383 daily ERA5 records for 2014–2025, obtained through the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). The requested coordinates were 6.9271 N, 79.8612 E; the returned grid cell was 7.0 N, 80.0 E. These are gridded estimates for the Colombo area, not station measurements. Source details, units and the request are in `data/provenance.json`.

Weather data by Open-Meteo, using ECMWF ERA5, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Daily data were converted to CSV and weather-history and seasonal features were derived. No personal data were collected.

Training recreates the saved model, metrics, held-out predictions and six figures in `results/plots`. It uses the included data snapshot; `python -m src.data` is only needed to collect a fresh extract, which may differ if the source is revised.

The Flask frontend groups the weather readings into expandable controls. A probability dial, original/edited comparison and SHAP bars update as readings change; recorded examples can be restored. All eleven contributions, full-size study charts, the raw dataset preview and CSV download are available. This is historical classification using delayed reanalysis, not a live forecast or flood warning.

The [report](https://github.com/PansiluAW/colombo-rain-study-ML/blob/main/report/269220E_ML_Report.pdf) covers the methodology, results and interpretation.
