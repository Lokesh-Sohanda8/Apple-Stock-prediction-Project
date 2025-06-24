# 📊 Apple Stock Price Forecasting Project

A comprehensive time series forecasting project using classic statistical models, machine learning algorithms, and deep learning (LSTM) to predict Apple's stock price for the next 30 business days.

---

## 🚀 Project Overview

This project aims to analyze, model, and forecast the future stock price of Apple Inc. (AAPL) using:

- 📈 **Time Series Decomposition & Stationarity Checks**
- 🧠 **Model Building with ARIMA, SARIMAX, Linear Regression, SVR, XGBoost, LightGBM, CatBoost, Random Forest, and LSTM**
- 🔍 **Evaluation Metrics: MSE, RMSE, MAE, MAPE**
- 📉 **Feature Engineering with Lag Values, Returns, Volatility, Moving Averages**
- 🌐 **Streamlit Web App for Real-time Forecasting**

---

## 📁 Contents

📁 notebooks/
└── Excel R Time Series Apple Forecast.ipynb
📁 streamlit_app/
└── app4.py (LSTM-powered UI)
📊 data/
└── Downloaded dynamically via yfinance
📝 README.md


---

## ⚙️ Models Compared

| Model             | Type              | Notes                            |
|------------------|-------------------|----------------------------------|
| ARIMA (1,0,1)     | Statistical       | Baseline univariate model        |
| SARIMAX           | Statistical       | Seasonal extension (non-seasonal here) |
| Linear Regression | ML                | Performed best overall 🔥        |
| SVR               | ML (kernel)       | Great with engineered features   |
| Random Forest     | Ensemble ML       | Handles non-linearity            |
| XGBoost           | Gradient Boosting | High-performance model           |
| LightGBM          | Gradient Boosting | Optimized for speed              |
| CatBoost          | Gradient Boosting | Stable with minimal tuning       |
| LSTM              | Deep Learning     | Captures temporal dependencies   |

---

## 🧪 Feature Engineering

- `Lag1`, `Lag2`, `Lag3` – to simulate memory for traditional ML
- `Returns` – to capture momentum (% change)
- `Volatility` – 21-day rolling standard deviation
- `MA10` / `MA30` – Short and medium-term moving averages

---

## 📊 Results

| Model               | MSE    | RMSE   | MAE    | MAPE (%) |
|--------------------|--------|--------|--------|----------|
| **Linear Regression** | ✅ Lowest overall | ✅ Fast, accurate |
| LSTM (Tuned)        | Lowest MAPE | ❌ Higher MSE due to overfitting |
| Others              | Competitive, but not consistent |

---

## 🌐 Streamlit App

Run the interactive forecasting UI:

```bash
streamlit run app4.py
