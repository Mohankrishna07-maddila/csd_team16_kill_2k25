# Step 1: Data preprocessing and model building
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load the dataset
df = pd.read_csv("Mobile price prediction\mobile_price_prediction_2.csv")

# Separate features and target
X = df.drop('Price (INR)', axis=1)
y = df['Price (INR)']

# Identify categorical and numerical columns
categorical_cols = ['Brand', 'Processor']
numerical_cols = ['RAM (GB)', 'Storage (GB)', 'Battery (mAh)']

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create and train the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the model pipeline and feature lists to pickle files
with open('mobile_price_model.pkl', 'wb') as model_file:
    pickle.dump(model_pipeline, model_file)

with open('model_columns.pkl', 'wb') as columns_file:
    pickle.dump({'numerical_cols': numerical_cols, 'categorical_cols': categorical_cols}, columns_file)

print("Model saved successfully!")
