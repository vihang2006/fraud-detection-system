# 🚀 Real-Time Fraud Detection System

A production-ready Machine Learning system for detecting fraudulent financial transactions in real time.

Built using FastAPI, Scikit-learn, SQLite, and a live monitoring dashboard.

---

## 📌 Features

- 🔍 Fraud prediction using Random Forest & Logistic Regression
- 📊 ROC-AUC performance comparison
- ⚡ Real-time transaction simulator
- 🗄 Transaction logging with SQLite database
- 📈 Live monitoring dashboard (Chart.js)
- 🚨 High-risk fraud alerts
- 🌐 REST API with FastAPI

---

## 🧠 Model Performance

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.9708 |
| Random Forest | 0.9774 |

Best Model Selected: **Random Forest**

---

## 🏗 System Architecture

Simulator → FastAPI Backend → ML Model → Database → Dashboard

---

## 🛠 Tech Stack

- Python
- FastAPI
- Scikit-learn
- SQLite
- Chart.js
- HTML / CSS
- Git

---

## ▶ How To Run Locally

### 1. Clone Repository

git clone https://github.com/vihang2006/fraud-detection-system.git
cd fraud-detection-system


### 2. Install Dependencies

pip install -r requirements.txt


### 3. Train Model

python backend/train_model.py


### 4. Start Backend

cd backend
python -m uvicorn main:app --reload


### 5. Start Simulator

python simulator.py


### 6. Open Dashboard

Open `frontend/index.html`

---

## 📈 Future Improvements

- Isolation Forest anomaly detection
- Docker containerization
- Deployment on cloud (Render / Railway)
- User authentication & admin panel
- Model drift monitoring

---

## 👨‍💻 Author

Vihang Bamnote
