from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
rf_model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame(data, index=[0])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = rf_model.predict(input_scaled)[0]
    risk_percentage = rf_model.predict_proba(input_scaled)[0, 1] * 100
    
    return jsonify({
        'prediction': int(prediction),
        'risk_percentage': float(risk_percentage)
    })

if __name__ == '__main__':
    app.run(debug=True)