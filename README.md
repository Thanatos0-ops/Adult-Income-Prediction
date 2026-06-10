# Adult Income Prediction

This project predicts whether a person's income is above or below $50K using the Adult Income dataset.

## Overview

The project includes a complete machine learning workflow:

- Data validation
- Data preprocessing
- Model training and evaluation
- Hyperparameter tuning
- Model export
- FastAPI prediction service
- Streamlit user interface

## Models Used

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- FastAPI
- Streamlit
- joblib

## Project Structure

```text
data/                 Raw dataset
models/               Saved model files and parameters
notebooks/            Exploratory analysis
src/data/             Data validation and preprocessing
src/models/           Training, tuning, and evaluation code
app.py                FastAPI backend
streamlit_app.py      Streamlit frontend
```

## How to Run

Train the model:

```bash
python -m src.models.train data/raw/adult.csv
```

Run the FastAPI backend:

```bash
uvicorn app:app --reload
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Output

The final trained pipeline is saved as:

```text
models/artifacts/income_classifier_pipeline.joblib
```
