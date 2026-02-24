# 🍎 Apple Stock Price Forecasting using Time Series Analysis

> A comprehensive data science project that analyzes historical Apple (AAPL) stock data from **2012 to 2019** using multiple time series and machine learning models to forecast prices and derive strategic business insights.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Team](#-team)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Workflow](#-workflow)
- [Models Used](#-models-used)
- [Model Performance](#-model-performance)
- [Final Model](#-final-model)
- [Deployment](#-deployment)
- [Business Insights](#-business-insights)
- [Challenges](#-challenges)
- [Future Work](#-future-work)

---

## 🧠 Project Overview

The primary objective of this project is to develop a predictive model that forecasts **Apple (AAPL) stock prices for the next 30 days** using historical data from 2012–2019. The solution helps investors, traders, and financial analysts make informed decisions based on stock trends and potential market movements.

**Key Goals:**
- Analyze historical AAPL stock data using time series techniques
- Build, evaluate, and compare multiple forecasting models
- Deploy an interactive Streamlit web app for real-time predictions
- Derive business insights to support strategic investment and operational planning

---

## 👨‍💻 Team

| Role | Name |
|------|------|
| **Mentor** | Mr. Dilawar Basha (Sir) |
| **Member** | Karedla Ganesh Reddy |
| **Member** | Amrutha A S |
| **Member** | Hritik Vivek Patil |
| **Member** | Nikhil Talapaneni |
| **Member** | O Lokesh |
| **Member** | Lokesh Laxman Sohanda |
| **Member** | Nisha Ashish Wandile |

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | Apple (AAPL) stock market data |
| **Time Span** | January 3, 2012 – December 30, 2019 |
| **Missing Values** | None |
| **Duplicates** | None |
| **Close Price Range** | $55.79 – $291.52 |
| **Volume Range** | 11.36M – 376.53M shares |

### Features

| Column | Description |
|--------|-------------|
| `Date` | Trading date (converted to DateTime) |
| `Open` | Opening price of AAPL for the day |
| `High` | Highest price reached during the day |
| `Low` | Lowest price reached during the day |
| `Close` | Closing price — **Target Variable** |
| `Adj Close` | Adjusted closing price |
| `Volume` | Number of shares traded |

**Files used:**
- `AAPL.csv` — Raw Apple stock data
- `AAPL_daily_2012_2019_cleaned.csv` — Cleaned dataset
- `Apple Stock Price History.csv` — Historical price reference
- `xgb_30day_forecast.csv` — XGBoost 30-day forecast output

---

## 📁 Project Structure

```
├── AAPL.csv
├── AAPL_daily_2012_2019_cleaned.csv
├── Apple Stock Price History.csv
├── Apple Stock Price Forecasting using Time S...   # Presentation
├── Excel R Time Series Apple Forecast.ipynb        # Main notebook
├── README.md
├── Requirement document.docx
├── Roadmap for 30 days.txt
├── aapl_candlestick.png
├── app4.py                                         # Streamlit app
├── model_accuracy_comparison_chart.json
├── model_comparison_results.csv
├── requirements2.txt
└── xgb_30day_forecast.csv
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Lokesh-Sohanda8/<repo-name>.git
cd <repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements2.txt
```

Key libraries used:
- `pandas`, `numpy` — Data manipulation
- `matplotlib`, `seaborn` — Visualization
- `statsmodels` — ARIMA, SARIMAX modeling
- `scikit-learn` — Linear Regression, SVR, Random Forest, preprocessing
- `xgboost`, `lightgbm`, `catboost` — Gradient boosting models
- `streamlit` — Web app deployment

### 3. Run the Jupyter Notebook

```bash
jupyter notebook "Excel R Time Series Apple Forecast.ipynb"
```

### 4. Launch the Streamlit App

```bash
streamlit run app4.py
```

---

## 🔄 Workflow

### Step 1 — Data Loading & Preprocessing
- Loaded AAPL stock data (2012–2019)
- Converted `Date` column to `datetime` format
- Checked and confirmed no missing values or duplicates
- Applied standard scaling and feature engineering (lagged prices, moving averages)

### Step 2 — Exploratory Data Analysis (EDA)
- Plotted line charts to observe the overall upward price trend
- Generated histograms, heatmaps, scatter plots, and pairplots
- Performed **Seasonal Decomposition** to identify Trend, Seasonality, and Residual components
- Analyzed **ACF and PACF plots** to identify autocorrelation structure
- Visualized a **Candlestick Chart** (2012–2019)

### Step 3 — Stationarity Testing
- Ran the **Augmented Dickey-Fuller (ADF) Test** → Initial p-value: `0.996` (non-stationary)
- Ran the **KPSS Test** for additional confirmation
- Applied **differencing** to make the series stationary → Post-differencing ADF p-value: `0.0` ✅

### Step 4 — Model Building
- Trained multiple models: Linear Regression, ARIMA, SARIMAX, XGBoost, Random Forest, SVR, LightGBM, CatBoost
- Used **lag features** and **time-based features** as predictors
- Applied **GridSearchCV** with **TimeSeriesSplit** for hyperparameter tuning

### Step 5 — Model Evaluation
- Compared all models using MSE, RMSE, MAE, and MAPE
- Selected **Linear Regression** as the final model

### Step 6 — Deployment
- Built and deployed an interactive **Streamlit web application**
- Features: date range selector, candlestick chart, 30-day price forecast, strategic insights

---

## 🏗️ Models Used

| Model | Type | Key Details |
|-------|------|-------------|
| **Linear Regression** | Statistical/ML | Baseline model; time as sole predictor |
| **ARIMA** | Statistical | Trend-based time series forecasting |
| **SARIMAX** | Statistical | Seasonal ARIMA with exogenous features; `(p,d,q)=(1,1,1)`, `(P,D,Q,s)=(1,1,0,12)` |
| **XGBoost** | Gradient Boosting | Lag features; `n_estimators=1000`, `max_depth=5`, `learning_rate=0.01` |
| **Random Forest** | Ensemble | Pattern recognition with multiple decision trees |
| **SVR** | Kernel-based | Support Vector Regression for non-linear patterns |
| **LightGBM** | Gradient Boosting | Fast gradient boosting with leaf-wise growth |
| **CatBoost** | Gradient Boosting | Handles categorical features natively |

---

## 📈 Model Performance

| Model | MSE | RMSE | MAE | MAPE |
|-------|-----|------|-----|------|
| **Linear Regression** ⭐ | **11.81** | **3.43** | **2.58** | 136.57 |
| CatBoost | 14.23 | 3.77 | 2.85 | 134.74 |
| LightGBM | 14.35 | 3.78 | 2.84 | 138.81 |
| SVR | 15.05 | 3.88 | 2.91 | 133.59 |
| Random Forest | 15.69 | 3.96 | 2.97 | 140.82 |
| XGBoost | 16.65 | 4.08 | 2.58 | 140.41 |
| ARIMA | 21.33 | 4.61 | 3.56 | 103.92 |
| SARIMAX | 21.33 | 4.61 | 3.56 | **99.61** |

> ⭐ **Linear Regression** achieved the best overall accuracy (lowest MSE, RMSE, MAE). **SARIMAX** achieved the lowest MAPE.

---

## 🎯 Final Model: Linear Regression

Linear Regression emerged as the best-performing model for this dataset. Key reasons:

- **Linear Trend Capture** — The dataset's stable upward price trend aligns perfectly with Linear Regression's strength in modeling linear relationships
- **No Overfitting** — With fewer parameters, it generalizes better than complex ensemble methods
- **Feature Suitability** — Engineered features (lagged prices, moving averages) exhibit strong linear correlations with the Close price
- **Effective Preprocessing** — Stationarity adjustments and scaling optimized the data for linear modeling
- **Consistent Error Minimization** — Outperforms all other models on MSE, RMSE, and MAE

**Best Metrics:**
- MSE: `11.8157`
- RMSE: `3.4374`
- MAE: `2.5888`

---

## 🚀 Deployment

The project is deployed as a **Streamlit web application** (`app4.py`) that provides:

- 📅 **Custom date range selector** for flexible forecasting
- 🕯️ **Candlestick chart** for historical price visualization
- 📈 **30-day price forecast** using the Linear Regression model
- 💡 **Strategic business insights** based on trend and volume analysis

To run locally:

```bash
streamlit run app4.py
```

---

## 💼 Business Insights

**Upward Trend for Strategic Planning** — Apple's consistent price growth from 2012–2019 reflects strong investor confidence, useful for projecting long-term growth and supporting product expansion strategies.

**Seasonality Supports Timing** — Clear yearly seasonal trends suggest businesses can time product launches, marketing campaigns, and inventory planning to align with favorable market cycles.

**Volatility Signals Risk Awareness** — Analysis of High/Low price ranges allows risk assessment and hedging strategy development.

**Volume Reflects Market Sentiment** — Trading volume spikes indicate major news or earnings events — a cue for syncing announcements with high investor engagement periods.

**Forecasting Supports Decision-Making** — Models like Linear Regression and SARIMAX provide predictive insights that support informed buy, sell, or hold decisions.

---

## 🧗 Challenges

- **Hyperparameter Tuning Complexity** — Fine-tuning SARIMAX, XGBoost, and SVR required extensive GridSearchCV experimentation, significantly increasing compute time
- **Stationarity Transformation** — Multiple differencing iterations and ADF diagnostic tests were needed before the series was suitable for modeling
- **Model Performance Trade-offs** — Balancing simplicity vs. flexibility across models with different strengths (e.g., SARIMAX's lower MAPE vs. Linear Regression's lower MSE/RMSE)
- **Seasonal Impact Interpretation** — Isolating seasonal trends from noise required careful decomposition analysis and domain knowledge

---

## 🔮 Future Work

- 🔄 **Enhanced Feature Engineering** — Integrate macroeconomic indicators (interest rates, inflation) and news sentiment scores
- 🧠 **Deep Learning Models** — Investigate LSTM or GRU networks for capturing long-term dependencies and nonlinear dynamics
- 📡 **Real-Time Forecasting** — Set up APIs and automated data pipelines for live predictions
- 🧪 **Scenario Simulations** — Build "what-if" analyses to evaluate potential shocks (earnings misses, policy changes)
- 🎯 **Multi-Asset Forecasting** — Apply the same framework to other stocks or sectors
- 📊 **Alternative Data Sources** — Leverage sentiment analysis from social media, financial news, and earnings transcripts

---

## 📄 License

This project was developed as part of an academic data science program. All data belongs to their respective owners.

---

<p align="center">
  Made with ❤️ by Team Apple Stock Forecasting | Mentor: Mr. Dilawar Basha
</p>
