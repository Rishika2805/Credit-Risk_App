import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import os

# Load DataSet
df = pd.read_csv("credit_risk_dataset.csv")


# Separate features and target variable
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Map categorical variables
X['cb_person_default_on_file'] = X['cb_person_default_on_file'].map({'Y': 1, 'N': 0})   

num_attribs = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
               'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                'cb_person_default_on_file' 
               ]

cat_attribs = ['person_home_ownership', 'loan_intent', 'loan_grade']



# Pipeline Function
def build_pipeline(num_attri, cat_attri):
    # Numerical Pipeline
    num_pipeline = Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical Pipeline
    cat_pipeline = Pipeline(
        [
            ("encoder", OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    # Full Pipeline
    full_pipeline = ColumnTransformer(
        [
            ('num', num_pipeline, num_attri),
            ('cat', cat_pipeline, cat_attri)
        ]
    )

    return full_pipeline


# Initialize the model
model = XGBClassifier(
    subsample=0.7,
 n_estimators=300,
 min_child_weight=1,
 max_depth=10,
 learning_rate=0.01,
 gamma=0.2,
 colsample_bytree=0.6
)

# Build the pipeline
pipeline = build_pipeline(num_attribs, cat_attribs)

# Prepare the data
X_prepared = pipeline.fit_transform(X)

# Train the model
model.fit(X_prepared, y)

# Save the model, pipeline and feature list
joblib.dump(model, 'credit_risk_model.pkl')
joblib.dump(pipeline, 'credit_risk_pipeline.pkl')
joblib.dump(X.columns.tolist(), 'credit_risk_features.pkl')


print("Model, pipeline and feature list have been saved successfully.")
