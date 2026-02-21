# Sri Lanka House Price Prediction (CatBoost + XAI)

This project trains a non-deep-learning model to predict Sri Lankan house prices from property features and provides explainability (SHAP, feature importance, PDP) plus a simple Streamlit UI.

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train model

```bash
C:/Python314/python.exe -m src.train --dataset dataset/cleaned_data_house_prices.csv --artifacts artifacts --trials 25
```

Artifacts written to `artifacts/`:
- `catboost_house_price.cbm`
- `model_meta.json`
- `test_true.npy`, `test_pred.npy`

### 3) Generate evaluation + explainability figures

```bash
C:/Python314/python.exe -m src.explain --dataset dataset/cleaned_data_house_prices.csv --artifacts artifacts --figures reports/figures
```

### 4) Run the UI

```bash
streamlit run app/streamlit_app.py
```

## Notes
- The dataset file is tab-separated (TSV) even though its extension is `.csv`.
- `Land size` contains some malformed values; `src/data.py` extracts the first numeric value and discards the rest.

## What to edit for your report
- Add the **true data source** you used to compile the dataset (website(s), API, manual collection, dates), and confirm ethical use.
- Define the meaning/units of `Land size` and `House size` (perches, sqft, etc.) if known.
