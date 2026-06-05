import pandas as pd
import sys
import json
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import validation modules
from src.data.data_validate import validate_schema
from src.data.base_transformations import preprocessor

PARAM_DISTRIBUTIONS = {
    'xgboost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    },
    'lightgbm': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5, -1],
        'classifier__num_leaves': [15, 31, 63],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0]
    }
}

def run_hyperparameter_tuning(data_path, model_choice='lightgbm'):
    if model_choice not in PARAM_DISTRIBUTIONS:
        raise ValueError(f"Unknow model choice: {model_choice}. Pick 'XGBOOST' or 'LIGHTGBM'.")
    
    # Load and Validate Data
    df = pd.read_csv(data_path)
    if not validate_schema(df):
        print("Data Failed Validation. Aborting Tuning")
        sys.exit(1)
    
    X = df.drop(columns=['income'])
    y = df['income'].map({'<=50K': 0, '>50K': 1})

    # Build the temporary pipeline track for the search
    from sklearn.pipeline import Pipeline
    if model_choice == 'xgboost':
        base_model = XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        base_model = LGBMClassifier(random_state=42, verbose=-1)
    
    search_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', base_model)
    ])

    # Configure Randomized Search CV
    print(f"---------Randomized Search CV for {model_choice}-----------")
    random_search = RandomizedSearchCV(
        estimator=search_pipeline,
        param_distributions= PARAM_DISTRIBUTIONS[model_choice],
        n_iter=10,                  # Sample 10 random combination
        scoring='f1',               # Target optimization metric
        cv=5,                       # 5 fold Stratified CV
        random_state=42,
        n_jobs=-1,                  # use all CPU cores
        verbose=1
    )

    random_search.fit(X, y)

    print("\n TUNING COMPLETE")
    print(f"Best 5 Fold Cross Validated F1 Score: {random_search.best_score_:.4f}")

    # Extract structural clean parameters
    best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
    print("Best Parameters discovered:", json.dumps(best_params, indent=4))

    # Save the parameters configuration to disk for train.py to look at later
    param_filename = f"models/best_params_{model_choice}.json"
    with open(param_filename, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved parameters configuration to {param_filename}")


if __name__ == "__main__":
    data_arg = sys.argv[1] if len(sys.argv) > 1 else "data/raw/adult.csv"
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "lightgbm"
    run_hyperparameter_tuning(data_arg, model_arg)