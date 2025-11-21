from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scalers
MODEL_FILE = "model/fraud_model.pkl"
SCALER_FILE = "model/scaler.pkl"
FEATURES_FILE = "model/features.pkl"

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
features = joblib.load(FEATURES_FILE)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract form data
    try:
        amount = float(data.get('amount', 0))
        merchant = data.get('merchant', 'other')
        time = data.get('time', 'morning')
        cardAge = float(data.get('cardAge', 0))
        frequency = float(data.get('frequency', 0))
        avgAmount = float(data.get('avgAmount', 0))
        international = 1 if data.get('international', 'no') == 'yes' else 0
    except Exception as e:
        return jsonify({"error": str(e)})

    # Create input array for model (make sure to match features order)
    input_data = []
    for f in features:
        if f == 'amount':
            input_data.append(amount)
        elif f == 'cardAge':
            input_data.append(cardAge)
        elif f == 'frequency':
            input_data.append(frequency)
        elif f == 'avgAmount':
            input_data.append(avgAmount)
        elif f == 'international':
            input_data.append(international)
        elif f.startswith('merchant_'):
            input_data.append(1 if f == f"merchant_{merchant}" else 0)
        elif f.startswith('time_'):
            input_data.append(1 if f == f"time_{time}" else 0)
        else:
            input_data.append(0)

    # Scale numerical features
    input_array = np.array([input_data])
    input_array[:, :4] = scaler.transform(input_array[:, :4])

    # Predict
    risk_prob = model.predict_proba(input_array)[0][1]  # probability of fraud
    risk_score = int(risk_prob * 100)
    is_fraud = risk_score > 50
    confidence = int(risk_score * 0.9 + 10)  # example confidence

    # Simple factor breakdown
    factors = {
        "amount": min(100, risk_score * 0.4),
        "location": min(100, risk_score * 0.2),
        "time": min(100, risk_score * 0.2),
        "pattern": min(100, risk_score * 0.2)
    }

    return jsonify({
        "isFraud": is_fraud,
        "riskScore": risk_score,
        "confidence": confidence,
        "factors": factors
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
