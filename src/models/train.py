import pandas as pd
import sys

# Import custom MLOps modules
from data_validate import validate_schema
from models.models import get_logistic_regression_pipeline, get_random_forest_pipeline
from models.evaluate import evaluate_pipeline, print_report

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
    print("Random Forest Classifer", rf_summary)

    if __name__ == "__main__":
        # Expect data path as an execution argument
        if len(sys.argv) > 1:
            run_training_pipeline(sys.argv[1])
        else:
            print("❌ Usage Error: Please provide the path to the data file. Example: python train.py data/adult.csv")
