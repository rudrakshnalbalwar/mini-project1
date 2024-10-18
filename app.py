from flask import Flask, request, jsonify, render_template # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore
import os
import traceback

app = Flask(__name__)

# Ensure the correct path to your model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'models', 'heart_model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'heart_scaler.joblib')

# Load the saved model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = [
            int(data['Age']), float(data['Sex']), float(data['ChestPainType']), 
            float(data['RestingBP']), float(data['Cholesterol']), float(data['FastingBS']),
            float(data['RestingECG']), float(data['MaxHR']), float(data['ExerciseAngina']),
            float(data['Oldpeak']), float(data['ST_Slope'])
        ]
        
        if model is None or scaler is None:
            raise Exception("Model or scaler not loaded properly")
        
        input_data_scaled = scaler.transform([input_data])
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)[0][1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability)
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)