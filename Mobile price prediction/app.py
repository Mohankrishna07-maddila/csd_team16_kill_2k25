# Step 2: Creating a Flask application
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and column information
with open('mobile_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model_columns.pkl', 'rb') as columns_file:
    column_info = pickle.load(columns_file)
    numerical_cols = column_info['numerical_cols']
    categorical_cols = column_info['categorical_cols']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        brand = request.form.get('brand')
        ram = float(request.form.get('ram'))
        storage = float(request.form.get('storage'))
        battery = float(request.form.get('battery'))
        processor = request.form.get('processor')
        
        # Create a dataframe with a single row for prediction
        input_data = pd.DataFrame({
            'Brand': [brand],
            'RAM (GB)': [ram],
            'Storage (GB)': [storage],
            'Battery (mAh)': [battery],
            'Processor': [processor]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', 
                              prediction=f"Predicted Price: ₹{prediction:.2f}",
                              input_data=f"Brand: {brand}, RAM: {ram}GB, Storage: {storage}GB, Battery: {battery}mAh, Processor: {processor}")
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)