import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

def evaluate_pipeline(pipeline, X, y, n_splits=5, random_state=42):
    """
    Validate a scikit-learn pipeline using Stratified K-Fold Cross-Validation.
    Ensure no data leakage occurs because the pipeline preprocessing
    is fitted strictly on the training folds.
    """
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Dictionaries to store mertices across fields

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc_score': []
    }

    print(f'Starting {n_splits}-Fold Stratified Cross-Validation...')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split the data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_val)

        # Check if model supports predict_proba for ROC_AUC
        if hasattr(pipeline, 'predict_probba'):
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            metrics['roc_auc_score'].append(roc_auc_score(y_val, y_proba))
        else:
            metrics['roc_auc_score'].append(np.nan)
        

        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        metrics['f1_score'].append(f1_score(y_val, y_pred, zero_division=0))

        print(f"Fold {fold} Complete. F1-Score: {metrics['f1_score'][-1]:.4f}")

        # Aggregate and return the final report summary
        summary = {metrics: {"mean": np.mean(values),  "std": np.std(values)} for metric, values in metrics.items()}
        return summary
    

def print_report(model_name, summary):
    print(f"\n=============={model_name} EVALUATION REPORT ==============")
    for metric, stats in summary.items():
        if np.isna(stats['mean']):
            print(f"{metric.upper() < 12} : N/A")
        else:
            print(f"{metric.upper(): < 12} : {stats['mean']:.4f} (+/- {stats['std']:.4f})")
        
    print("=======================================================\n")