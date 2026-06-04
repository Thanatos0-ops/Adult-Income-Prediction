from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import the global preprocessor
from src.data.base_transformations import preprocessor

def get_logistic_regression_pipeline(): 
    return Pipeline([
        ('preprocessor', preprocessor),
        ('scalar', StandardScaler(with_mean=False)), # with_mean=False handles sparse/one-hot matrices cleanly
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

def get_random_forest_pipeline():
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifer', RandomForestClassifier(random_state=42, n_jobs=-1)) # n_jobs controls how many CPU cores are use in parallel for operations such as training,(1 = one core, 2 = two cores, -1 = all availabe cores)
    ])
