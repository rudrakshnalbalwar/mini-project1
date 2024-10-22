import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# Build the ANN model
ann_model = Sequential()
ann_model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
ann_model.add(BatchNormalization())
ann_model.add(Dropout(0.2))

ann_model.add(Dense(32, activation='relu'))
ann_model.add(BatchNormalization())
ann_model.add(Dropout(0.2))

ann_model.add(Dense(16, activation='relu'))
ann_model.add(BatchNormalization())
ann_model.add(Dropout(0.2))

ann_model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=0.0001)

# Train the model
ann_model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, validation_data=(X_test_scaled, y_test), 
              callbacks=[early_stopping, reduce_lr], verbose=1)

# Evaluate the model
y_pred = (ann_model.predict(X_test_scaled) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
ann_model.save('ann_model.h5')
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
    prediction = (ann_model.predict(final_features_scaled) > 0.5).astype("int32")
    risk_percentage = ann_model.predict(final_features_scaled)[0][0] * 100

    output = round(prediction[0][0], 2)
    result = f'The patient {"has" if output == 1 else "does not have"} heart disease.'
    risk = f'Risk percentage: {risk_percentage:.2f}%'
    
    print("Prediction:", result)
    print("Risk:", risk)
    
    return render_template('index.html', prediction_text=result, risk_percentage=risk)

if __name__ == "__main__":
    app.run(debug=True)