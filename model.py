import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from flask import Flask, request, jsonify, render_template

# Load and preprocess the data
data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features_scaled = scaler.transform(final_features)
    prediction = rf_model.predict(final_features_scaled)
    risk_percentage = rf_model.predict_proba(final_features_scaled)[0][1] * 100

    output = round(prediction[0], 2)
    result = f'The patient {"has" if output == 1 else "does not have"} heart disease.'
    risk = f'Risk percentage: {risk_percentage:.2f}%'
    
    print("Prediction:", result)
    print("Risk:", risk)
    
    return render_template('index.html', prediction_text=result, risk_percentage=risk)

if __name__ == "__main__":
    app.run(debug=True)