from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
rf_model = joblib.load('rf_model.joblib')  # Load the trained RandomForest model
scaler = joblib.load('scaler.joblib')      # Load the trained scaler

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the POST request
    data = request.get_json()
    input_data = pd.DataFrame(data, index=[0])
    
    # Scale the input data using the pre-trained scaler
    input_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = rf_model.predict(input_scaled)[0]
    
    # Get the probability of the positive class (heart disease risk)
    risk_percentage = rf_model.predict_proba(input_scaled)[0, 1] * 100
    
    # Return the result as a JSON response
    return jsonify({
        'prediction': int(prediction),              # Prediction (0 or 1)
        'risk_percentage': float(risk_percentage)   # Probability in percentage
    })

if __name__ == '__main__':
    app.run(debug=True)
