# 🎓 Student Grade Predictor

A machine learning web app that predicts students' final grades (G3) using 4 different ML models — built with Python and deployed with Streamlit.

🔗 **Live Demo:** [student-grade-predictor-9icckmcpwrlkhxufugxfri.streamlit.app](https://student-grade-predictor-9icckmcpwrlkhxufugxfri.streamlit.app/)

---

## 📌 Overview

This project trains and compares 4 regression models on the Student Performance Dataset to predict the final grade G3 based on student demographics, study habits, and social factors.

---

## 🤖 Models

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model — fast and interpretable |
| Random Forest | Ensemble of 200 decision trees |
| SVR | Support Vector Regressor with RBF kernel |
| XGBoost | Gradient boosting — focuses on correcting residual errors |

---

## 📊 App Pages

- **🏠 Overview** — Dataset summary and model descriptions
- **📊 Model Comparison** — RMSE, MAE, R² comparison across all 4 models
- **🔮 Predict Grade** — Input student features and get instant predictions
- **📈 Feature Importance** — Top features for Random Forest and XGBoost

---

## 📓 Notebook

The full training and analysis notebook is available in ML Project For UNI.ipynb — includes EDA, preprocessing, model training, evaluation, and comparison for all 4 models.

---

## 🚀 Run Locally

git clone https://github.com/zyadragab/Student-Grade-Predictor
cd Student-Grade-Predictor
pip install -r requirements.txt
streamlit run app.py

---

## 📁 Project Structure

├── app.py                       # Main Streamlit app
├── ML Project For UNI.ipynb     # Full training notebook
├── requirements.txt             # Dependencies
├── student_data.csv             # Dataset
├── models/                      # Saved trained models
│   ├── lr.pkl
│   ├── rf.pkl
│   ├── svr.pkl
│   ├── xgb.pkl
│   └── scaler.pkl
└── README.md
