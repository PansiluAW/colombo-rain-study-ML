# colombo-rain-study-ML
Assignment targeting the applied machine learning concepts attended during the MSc program at the University of Moratuwa.

## Colombo, tomorrow.

Next-day rain classification for a Colombo-area ERA5 grid cell. MSc in Artificial Intelligence assignment by Pansilu Wijesiri, 269220E.

The Streamlit interface follows a simple weather-notebook layout, drawing on the text-first presentation of [wttr.in](https://wttr.in/). Users enter weather from the completed day before the target date. The saved regularised Quadratic Discriminant Analysis (QDA) model returns a wet/dry classification, a probability and exact interventional SHAP contributions for the actual entry.

## Run

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

On Windows activate with `.venv\Scripts\activate`. The saved model is included; training is not required to run the app. No API key is needed.

## Reproduce the experiment

```bash
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.explain
python -m unittest discover -s tests -p 'test_*.py'
```

The fixed source snapshot is included. Run `python -m src.data` only to collect a new snapshot, since upstream revisions may change the values. The original request and checksum are in `data/provenance.json`.

## Data and method

The dataset contains 4,383 daily records for 2014–2025 from the Open-Meteo Historical Weather API, explicitly requesting ECMWF ERA5 and the Asia/Colombo time zone. The requested coordinates were 6.9271 N, 79.8612 E; the returned grid location was 7.0 N, 80.0 E. These are gridded reanalysis estimates for the Colombo area, not station measurements.

A wet day means precipitation of at least 1 mm on the following day. Removing incomplete seven-day windows and the final unavailable target leaves 4,376 rows. Eleven numerical predictors describe preceding-day weather, trailing rain means, pressure change and target-date season.

| Partition | Target dates | Rows |
|---|---|---:|
| Training | 2014-01-08 to 2021-12-31 | 2,915 |
| Validation | 2022-01-01 to 2023-12-31 | 730 |
| Test | 2024-01-01 to 2025-12-31 | 731 |

QDA is outside the supplied lecture list (linear/logistic regression, decision trees, random forest, SVM, KNN, XGBoost, AdaBoost, gradient boosting, LightGBM and CatBoost). It is a Gaussian generative classifier with separate class covariances and a quadratic decision boundary; it uses no deep learning. Logistic regression is included only as a comparison baseline.

The three rainfall features receive `log1p`; all 11 inputs are then standardised using training statistics. Eight covariance regularisation settings were compared on validation AUC. The selected `reg_param` is 0.75 and the validation-selected threshold is 0.78. Class priors are training class frequencies. The model was not refitted using validation or test rows.

Test AUC is 0.872765; accuracy 0.813953; balanced accuracy 0.785678; F1 0.872897; Brier loss 0.131541. The confusion matrix is `[[128, 47], [89, 467]]` with actual dry/wet rows and predicted dry/wet columns. Logistic regression has higher AUC and lower Brier loss. Persistence has higher accuracy and F1, and monthly climatology has higher balanced accuracy. Full comparisons are in `results/metrics.json`.

Exact interventional SHAP explains the QDA log-odds polynomial using the empirical background of all 2,915 training rows. The implementation allocates linear, squared and pairwise terms analytically. An independent test enumerates all 2,048 feature coalitions against 32 background rows and checks the Shapley-weighted result. Contributions reconstruct the raw score; their baseline is mean training log odds, not mean probability. Replacement coalitions can break correlations and the sine/cosine relationship, so these are model associations, not causal effects.

ERA5 arrives after the observation day. This is a retrospective classification study, not an operational forecast or flood warning. Probabilities have not been independently calibrated. Rainfall remains non-Gaussian even after transformation, which limits QDA's density assumptions.

## Project files

- `streamlit_app.py`: interface, prediction and explanation display.
- `src/`: data collection, preprocessing, training, evaluation, XAI and shared input construction.
- `data/`: raw snapshot, CSV, derived modelling data and provenance.
- `models/qda.json`: portable quadratic coefficients, preprocessing statistics and SHAP background moments.
- `src/model.py`: QDA score reconstruction and analytic SHAP.
- `results/`: metrics, predictions, SHAP outputs and figures.
- `tests/`: temporal alignment, input validation and saved-prediction checks.
- `.streamlit/config.toml`: the app's theme and runtime configuration.

## Deploy from GitHub

1. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in and connect your GitHub account if requested.
2. Choose **Create app**, then **Yup, I have an app**.
3. Enter the settings below. If the repository is not suggested, enter it manually or use **Paste GitHub URL** with the link to `streamlit_app.py` in this repository.

| Setting | Value |
|---|---|
| Repository | `PansiluAW/colombo-rain-study-ML` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python version, under Advanced settings | `3.12` |
| Secrets | Leave empty |

4. Save the advanced settings and choose **Deploy**. Dependencies are installed from the root `requirements.txt`; the saved model and data are already included. No training run or API key is needed.
5. Once the app opens, select **04 Jan 2025 / correct dry day**. The expected result is **DRY**, with a **29.6%** model probability. Change humidity from **76% to 86%** and submit: the probability should become **62.0%**, still below the **78%** wet-day cutoff, and the explanation should update. Set minimum temperature above maximum temperature to check that an error replaces the prediction.
6. Check **Settings > Sharing** and confirm the examiner can view the app. Test the copied app link in a private/incognito window. Changes pushed to `main` update the deployed app.

Deployment and sharing instructions follow the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [sharing guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## Demo video

Record 3–5 minutes showing this QDA frontend: enter weather, inspect a wet and dry example, change an input, explain the SHAP chart, show validation, and open the results and data tabs. Include the GitHub repository and deployed app link. Mention the retrospective ERA5 limitation and one error example. Check the final recording against the current model and results before submission.

The report PDF, source/dataset package and 3–5 minute demo video are the assignment submission items. Check examiner access to any hosted links before submitting.

## Data attribution

Weather data by [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api), using ECMWF ERA5, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Daily JSON was converted to CSV; lagged, rolling and seasonal predictors were derived. The source does not endorse the model. No personal data were collected.
