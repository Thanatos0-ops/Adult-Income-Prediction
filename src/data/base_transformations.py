import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer


# ------------------------Custom Scikit-learn Transformers-------------------

class QuestionMarkCleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self, value="?"):
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.replace(self.value, np.nan)
        

class RareLabelEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        self.frequent_labels_ = {}

        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.frequent_labels_[col] = freq[freq >= self.threshold].index.tolist()
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            if col in self.frequent_labels_:
                X[col] = np.where(
                    X[col].isin(self.frequent_labels_[col]),
                    X[col],
                    'Others'
                )
        return X
        

# ----------------------COLUMN GROUPS------------------------------

numeric_features = [
    'age',
    'education.num',
    'hours.per.week'
]

log_features = [
    'capital.gain',
    'capital.loss'
]

categorical_missing_constant = [
    'workclass',
    'occupation'
]

categorical_missing_mode = [
    'native.country'
]


other_categorical = [
    'marital.status',
    'relationship',
    'race',
    'sex'
]

categorical_features = (
    categorical_missing_constant +
    categorical_missing_mode +
    other_categorical
)


# -------------------------Numerical Sub-Pipelines----------------------------

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Log Transform Pipeline
log_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),

    ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'))
])


# ----------------------------------Categorical Sub-Pipelines--------------------

constant_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('rare', RareLabelEncoder(threshold=0.01)),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

mode_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('rare', RareLabelEncoder(threshold=0.01)),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


# ------------------------------Master Preprocessor------------------------

feature_transformer = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('log', log_pipeline, log_features),
    ('cat_constant', constant_categorical_pipeline, categorical_missing_constant),
    ('cat_mode', mode_categorical_pipeline, categorical_missing_mode),
    ('cat_other', mode_categorical_pipeline, other_categorical)
], remainder='drop')


# -------------------------------Global Pipeline-----------------------

preprocessor = Pipeline([

    # 1. Global Dataframe Cleaning step
    ('clean_missing', QuestionMarkCleaner()),

    # 2. Complete processing down to array formatting
    ('features', feature_transformer)
])