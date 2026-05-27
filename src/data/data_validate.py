import sys
import pandas as pd
import numpy as np

# Define expected schema and types based on eda
EXPECTED_SCHEMA = {
    "age": "int64",
    "workclass":"object",
    "fnlwgt":"int64", 
    "education":"object",
    "education.num":"int64", 
    "marital.status":"object",
    "occupation":"object",
    "relationship":"object",
    "race":"object",
    "sex":"object",
    "capital.gain":"int64", 
    "capital.loss":"int64", 
    "hours.per.week":"int64", 
    "native.country":"object",
    "income":"object"
}

def validate_schema(df):
    """
    Validates the schema of the DataFrame against the expected schema, ranges, and constraints.
    Returns True if validation passes, otherwise raises an error with details.
    """
    print("Validating schema...")

    validation_passed = True


    # Check for missing columns
    missing_columns = [col for col in EXPECTED_SCHEMA.keys() if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        validation_passed = False
    

    # Check for unexpected columns
    unexpected_columns = [col for col in df.columns if col not in EXPECTED_SCHEMA.keys()]
    if unexpected_columns:
        print(f"WARNING: Unexpected columns found: {unexpected_columns}")


    # Check data types
    for col, expected_type in EXPECTED_SCHEMA.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                print(f"Error: Column '{col}' has type '{actual_type}' but expected '{expected_type}'")
                validation_passed = False


    # Check value ranges (Continuous Features)
    if (df["age"] <= 0).any():
        print("WARNING: Dataset contains negative or zero values for 'age'.")
        validation_passed = False
    
    if (df["hours.per.week"] < 1).any() or (df["hours.per.week"] > 168).any():
        print("VALUE ERROR: 'hours.per.week' contains values outside realistic bounds (1-168).")
        validation_passed = False
    
    if (df["capital.gain"] < 0).any() or (df["capital.loss"] < 0).any():
        print("VALUE ERROR: Financial features cannot contain negative values.")

    
    # Check Targe Feature Values
    valid_targets = {"<=50K", ">50K", 0, 1}
    unique_targets = set(df['income'].unique())
    if not unique_targets.issubset(valid_targets):
        print("TARGET ERROR: Unexpected classes found in target variable 'income' : {unique_targets - valid_targets}")


    if validation_passed:
        print("Data Validation Successful! Dataset matches required schema and constraints.")
    else:
        print("Data Validation Failed. Review the errors above.")
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        try:
            data = pd.read_csv("data_path")
            if not validate_schema(data):
                sys.exit(1)
        except Exception as e:
            print(f"Failed to read data at {data_path}. Error: {e}")
            sys.exit(1)