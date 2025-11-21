import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/fraud_model.pkl')
scaler = joblib.load('model/scaler.pkl')
features = joblib.load('model/features.pkl')  # order of features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Prepare input array in the same order as features
    X_input = [float(data.get(f, 0)) for f in features]
    X_scaled = scaler.transform([X_input])
    
    pred = model.predict(X_scaled)[0]
    risk_score = float(model.predict_proba(X_scaled)[0][1]) * 100
    
    return jsonify({
        "isFraud": bool(pred),
        "riskScore": round(risk_score, 2),
        "confidence": round(risk_score, 2)  # optional
    })
