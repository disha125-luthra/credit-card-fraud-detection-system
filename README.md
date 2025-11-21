# Credit Card Fraud Detection System

A machine learning–based web application that detects fraudulent credit card transactions in real time.  
This project uses a trained classification model and a Flask backend to provide instant predictions through a simple web interface.

---

## 🚀 Features
- Predicts whether a transaction is **Fraudulent** or **Legitimate**
- Built with **Flask**, **scikit-learn**, and **joblib**
- Clean web interface using **HTML/CSS**
- Pre-trained ML model loaded for fast inference
- Easy to deploy on **Render**, **Heroku**, or **local machine**

---

## 📂 Project Structure
credit-card-fraud-detection-system/
│
├── dataset/
│   ├── credit.csv
│   ├── credit_preprocessed.csv
│
├── model/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── preprocess.py
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
└── venv/ (ignored)

How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/disha125-luthra/credit-card-fraud-detection-system.git
cd credit-card-fraud-detection-system

2️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Preprocess Data
python3 preprocess.py

5️⃣ Train the Model
python3 train.py

6️⃣ Run the Web App
python3 app.py


App runs on:
👉 http://127.0.0.1:5000

🧠 Model Features

Uses scaled numerical features

Stored in features.pkl

Machine learning model stored as fraud_model.pkl

Scaler saved as scaler.pkl

📊 Evaluation

Run:

python3 evaluate.py


Generates:

Accuracy

Precision

Recall

AUC score

🌐 Web Interface

Simple Flask UI where users can:

Enter transaction values 
Get real-time fraud prediction


