import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import joblib # type: ignore

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv('heart.csv')
    
    # Encode categorical variables
    data['Sex'] = data['Sex'].map({"M": 1, "F": 0})
    data['ChestPainType'] = data['ChestPainType'].map({"TA": 1, "ATA": 2, "NAP": 3, "ASY": 4})
    data['RestingECG'] = data['RestingECG'].map({"Normal": 1, "ST": 2, "LVH": 3})
    data['ExerciseAngina'] = data['ExerciseAngina'].map({"N": 0, "Y": 1})
    data['ST_Slope'] = data['ST_Slope'].map({"Up": 1, "Flat": 2, "Down": 3})
    
    return data

# Split data and train model
def train_model(data):
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return model, scaler

# Save model and scaler
def save_model_and_scaler(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

# Prediction function
def predict_heart_disease(model, scaler, input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]
    return prediction[0], probability

if __name__ == "__main__":
    data = load_and_preprocess_data('heart.csv')
    model, scaler = train_model(data)
    save_model_and_scaler(model, scaler, 'heart_model.joblib', 'heart_scaler.joblib')