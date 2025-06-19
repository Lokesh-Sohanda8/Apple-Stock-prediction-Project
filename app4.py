import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

# Mute warnings for a cleaner output
warnings.filterwarnings('ignore')

# Custom CSS for UI styling
st.markdown("""
<style>
body {
    background-color: #0a0f1a;
    color: #e0f7fa;
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0f1a, #1a2a44);
}
.title {
    text-align: center;
    font-size: 2.8em;
    color: #00ffcc;
    text-shadow: 0 0 20px #00ffcc, 0 0 10px #00cc99;
    margin-bottom: 35px;
    font-weight: 700;
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 10px #00ffcc; }
    to { text-shadow: 0 0 20px #00cc99, 0 0 30px #00ffcc; }
}
.sidebar .sidebar-content {
    background: #1a2a44;
    border-right: 2px solid #00ffcc;
}
.sidebar .stDateInput > div > div {
    background: #2a4060;
    border: 2px solid #00ffcc;
    border-radius: 10px;
    color: #e0f7fa;
}
.sidebar .stDateInput > div > div > input {
    color: #e0f7fa;
}
.container {
    background: rgba(255, 255, 255, 0.1);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 6px 40px rgba(0, 255, 204, 0.2);
    margin-bottom: 30px;
    transition: transform 0.4s ease, box-shadow 0.4s ease;
}
.container:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 50px rgba(0, 255, 204, 0.3);
}
.chart-container {
    background: rgba(26, 42, 68, 0.7);
    padding: 25px;
    border-radius: 12px;
    margin-top: 25px;
    transition: background 0.3s ease;
}
.chart-container:hover {
    background: rgba(26, 42, 68, 0.9);
}
.table-container {
    overflow-x: auto;
    background: rgba(26, 42, 68, 0.7);
    padding: 25px;
    border-radius: 12px;
    margin-top: 25px;
}
.insights {
    background: rgba(0, 255, 204, 0.15);
    padding: 25px;
    border-left: 8px solid #00ffcc;
    border-radius: 12px;
    margin-top: 30px;
    font-size: 1.2em;
    line-height: 1.6;
}
.insights li {
    margin-bottom: 15px;
    color: #e0f7fa;
}
.stButton>button {
    background-color: #00ffcc;
    color: #0a0f1a;
    border: none;
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 600;
    transition: background-color 0.3s ease, transform 0.3s ease;
}
.stButton>button:hover {
    background-color: #00cc99;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Apple 30-Day Stock Prediction!</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        stock_data = yf.download('AAPL', start='2011-01-01', end='2020-12-31', progress=False)
        if stock_data.empty:
            raise ValueError("No data retrieved from yfinance!")
        return stock_data
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None

def create_dataset(data, lag=5):
    if len(data) < lag + 1:
        return None, None, None, None, None
    df = data[['Close']].copy()
    for i in range(1, lag + 1):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    if df.empty:
        return None, None, None, None, None
    X = df[[f'Close_lag_{i}' for i in range(1, lag + 1)]].values
    y = df['Close'].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    return X_scaled, y_scaled, scaler_X, scaler_y, df.index

@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_30_days(model, scaler_X, scaler_y, data, start_date, lag=5):
    last_data = data['Close'].tail(lag).values[::-1].reshape(1, -1)
    last_data_scaled = scaler_X.transform(last_data)
    predictions = []
    current_input = last_data_scaled.copy()
    for _ in range(30):
        pred_scaled = model.predict(current_input)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        predictions.append(pred)
        current_input = np.roll(current_input, -1)
        current_input[0, -1] = pred_scaled[0]
        current_input = scaler_X.transform(scaler_y.inverse_transform(current_input).reshape(1, -1))
    dates = pd.date_range(start=start_date + timedelta(days=1), periods=30, freq='B')
    return pd.DataFrame({'Date': dates, 'Predicted Close': predictions})

data = load_data()
if data is None:
    st.stop()

with st.sidebar:
    st.markdown('<h3 style="color:#00ffcc; text-align:center;">Date Selector</h3>', unsafe_allow_html=True)
    start_date = st.date_input("Pick Your Start Date", min_value=datetime(2011, 1, 1), max_value=datetime(2020, 12, 31), value=datetime(2019, 1, 1))
    end_date = st.date_input("Pick Your End Date", min_value=start_date, max_value=datetime(2020, 12, 31), value=start_date + timedelta(days=30))
    if st.button("🚀 Generate Predictions"):
        st.session_state.predictions_generated = True

if 'predictions_generated' not in st.session_state:
    st.session_state.predictions_generated = False

with st.container():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    if st.session_state.predictions_generated and start_date <= end_date:
        min_date = data.index.min().date()
        if start_date < min_date + timedelta(days=5):
            st.error(f"Start date must be after {min_date + timedelta(days=5):%Y-%m-%d} to allow for lag features!")
        else:
            filtered_data = data.loc[start_date:end_date].copy()
            if not filtered_data.empty:
                historical_data = data.loc[:start_date]
                X, y, scaler_X, scaler_y, indices = create_dataset(historical_data)
                if X is None:
                    st.error("Insufficient data before start date to create lagged features!")
                else:
                    model = train_model(X, y)
                    pred_df = predict_30_days(model, scaler_X, scaler_y, historical_data, start_date)

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
                    fig.patch.set_facecolor('#1a2a44')
                    ax1.set_facecolor('#1a2a44')
                    ax2.set_facecolor('#1a2a44')

                    dates = mdates.date2num(filtered_data.index)
                    for i in range(len(filtered_data)):
                        x = dates[i]
                        open_price = float(filtered_data['Open'].iloc[i])
                        high_price = float(filtered_data['High'].iloc[i])
                        low_price = float(filtered_data['Low'].iloc[i])
                        close_price = float(filtered_data['Close'].iloc[i])
                        if np.any(np.isnan([open_price, high_price, low_price, close_price])):
                            continue
                        if close_price >= open_price:
                            ax1.bar(x, close_price - open_price, bottom=open_price, color='#00ffcc', width=0.6, align='center')
                            ax1.vlines(x, low_price, open_price, color='#00ffcc', linewidth=1)
                            ax1.vlines(x, close_price, high_price, color='#00ffcc', linewidth=1)
                        else:
                            ax1.bar(x, open_price - close_price, bottom=close_price, color='#ff6b6b', width=0.6, align='center')
                            ax1.vlines(x, low_price, close_price, color='#ff6b6b', linewidth=1)
                            ax1.vlines(x, open_price, high_price, color='#ff6b6b', linewidth=1)

                    ax1.xaxis_date()
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
                    ax1.set_title('Candlestick Chart', color='#e0f7fa', fontsize=14)
                    ax1.set_ylabel('Price ($)', color='#e0f7fa')
                    ax1.grid(True, linestyle='--', alpha=0.3, color='#e0f7fa')
                    ax1.tick_params(colors='#e0f7fa')

                    ax2.plot(pred_df['Date'], pred_df['Predicted Close'], label='Predicted', color='#00cc99', linewidth=2, marker='o')
                    ax2.set_title('Predicted Close Price', color='#e0f7fa', fontsize=14)
                    ax2.set_ylabel('Price ($)', color='#e0f7fa')
                    ax2.legend(facecolor='#1a2a44', edgecolor='#00ffcc', labelcolor='#e0f7fa')
                    ax2.grid(True, linestyle='--', alpha=0.3, color='#e0f7fa')
                    ax2.tick_params(colors='#e0f7fa')

                    plt.xlabel('Date', color='#e0f7fa')
                    plt.tight_layout()

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="table-container">', unsafe_allow_html=True)
                    st.write("### Predicted Values Showcase")
                    st.dataframe(pred_df.style.format({'Predicted Close': '{:.2f}', 'Date': '{:%Y-%m-%d}'}))
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="insights">', unsafe_allow_html=True)
                    st.write("### Strategic Insights for Apple Inc.")
                    start_price = pred_df['Predicted Close'].iloc[0]
                    end_price = pred_df['Predicted Close'].iloc[-1]
                    price_increase = end_price - start_price
                    price_increase_pct = (price_increase / start_price) * 100 if start_price != 0 else 0
                    trend = "upward" if price_increase > 0 else "downward"
                    volatility = pred_df['Predicted Close'].std()
                    investment_strategy = "cautious divestment" if trend.lower() == "downward" else "aggressive investment"

                    pred_df['Date'] = pd.to_datetime(pred_df['Date'], errors='coerce')
                    if pd.notnull(pred_df['Date'].iloc[0]) and pd.notnull(pred_df['Date'].iloc[-1]):
                        start_date_str = pred_df['Date'].iloc[0].strftime('%Y-%m-%d')
                        end_date_str = pred_df['Date'].iloc[-1].strftime('%Y-%m-%d')
                    else:
                        start_date_str = "N/A"
                        end_date_str = "N/A"

                    st.markdown(f"""
                    - **Market Trend Alert**: An {trend} trend is forecasted, signaling a {trend} market movement — consider **{investment_strategy}** strategies.

                    - **Price Snapshot**:
                        - **Start**: ${start_price:.2f} on {start_date_str}
                        - **Ending**: ${end_price:.2f} on {end_date_str}
                    

                    - **Profit Potential**: Expect a ${price_increase:.2f} increase ({price_increase_pct:.2f}% change) — optimize stock allocation accordingly.

                    - **Volatility Check**: With a standard deviation of {volatility:.2f}, {'high volatility' if volatility > 5 else 'stable conditions'} suggest {'hedging risks' if volatility > 5 else 'confident expansion'}.

                    - **Candlestick Clue**: Green candles signal price gains, red candles show losses — use these trends to time your market moves with predictions!
                    """)

                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("No historical data for the selected date range!")
    elif st.session_state.predictions_generated and start_date > end_date:
        st.error("End date must be after start date!")

    st.markdown('</div>', unsafe_allow_html=True)
