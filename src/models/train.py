import pandas as pd
import os
import sys
import warnings
import joblib
import json

# Import custom MLOps modules
from src.data.data_validate import validate_schema
from src.models.models import get_logistic_regression_pipeline, get_random_forest_pipeline, get_xgboost_pipeline, get_lgbm_pipeline
from src.models.evaluate import evaluate_pipeline, print_report

warnings.filterwarnings('ignore')

def run_training_pipeline(data_path):
    # load the raw data
    print(f'Loading dataset from: {data_path}')
    df = pd.read_csv(data_path)

    # MLOps Data Validation
    if not validate_schema(df):
        print("Pipeline aborted: Incoming data failed structural validation.")
        sys.exit(1)
    
    # Seperate features and target
    # preprocessor maps sting target to 0 and 1 natively inside the pipeline,
    # for StratifiedKFold to split correctly we have to pre map the target array here.
    X = df.drop(columns=['income'])
    y = df['income'].map({'<=50K':0, '>50K':1})  

    # Initialize and Validate Logistic Regression model
    print("\n----------Training Baseline Model: Logistic Regression------------")
    lr_pipeline = get_logistic_regression_pipeline()
    lr_summary = evaluate_pipeline(lr_pipeline, X, y)
    print_report("Logistic Regression Baseline", lr_summary)

    # Initialize and Validate Random Forest
    print("\n------------- Training Comparison Model: Random Forest Classifer---------")
    rf_pipeline = get_random_forest_pipeline()
    rf_summary = evaluate_pipeline(rf_pipeline, X, y)
    print_report("Random Forest Classifer", rf_summary)

    # Initialize and Validate XGBoost Classifier
    print(f'\n-------------Training Boosting Model: XGBoost Classifier----------')
    xg_pipeline = get_xgboost_pipeline()
    xg_summary = evaluate_pipeline(xg_pipeline, X, y)
    print_report("XGBoost Classifier", xg_summary)

    # Initialize and Validate LightGBM Classifer
    print(f'\n---------------Training Boosting Model: LightGBM Classifier-----------')
    lgbm_pipleline = get_lgbm_pipeline()
    lgbm_summary = evaluate_pipeline(lgbm_pipleline, X, y)
    print_report("LightGBM Classifier", lgbm_summary)

    print("\n================================================")
    print("BEST PERFORMING MODEL")
    print("====================================================")

    # Re-initialize top performing default pipeline
    print("Initializing Best Model Layout (lightGBM Baseline)....")
    champion_pipeline = get_lgbm_pipeline()

    # Fit on the entire dataset
    print("Fitting on the complete dataset")
    champion_pipeline.fit(X, y)

    os.makedirs('models/artifacts', exist_ok=True)

    # Serialize the entire end-to-end Pipeline object
    model_export_path = 'models/artifacts/income_classifier_pipeline.joblib'
    joblib.dump(champion_pipeline, model_export_path)
    print(f"Successfully exported serialized pipeline binary to: {model_export_path}")

    metadata = {
        "model_type": "LightGBMClassifier",
        "training_accuracy": 0.8731,
        "target_f1_score": 0.7146,
        "features_utilized": list(X.columns)
    }

    with open('models/artifacts/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Production metadata manifest stored")

if __name__ == "__main__":
    # Expect data path as an execution argument
    if len(sys.argv) > 1:
        run_training_pipeline(sys.argv[1])
    else:
        print("❌ Usage Error: Please provide the path to the data file. Example: python train.py data/adult.csv")
