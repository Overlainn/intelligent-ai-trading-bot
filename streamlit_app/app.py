import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time

# ✅ Add trading_engine folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import your strategy logic
from trading_engine.strategy import should_enter_trade

# Mock function to simulate price data
def get_mock_data():
    np.random.seed(0)
    prices = np.cumsum(np.random.randn(100)) + 100
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min'),
        'price': prices
    })
    return df

# Streamlit UI
st.set_page_config(page_title="Intelligent AI Trading Bot", layout="wide")
st.title("📈 Intelligent AI Trading Bot")

# Load data
df = get_mock_data()
st.line_chart(df.set_index("timestamp")["price"])

# Decision block
latest_price = df["price"].iloc[-1]
should_trade, signal = should_enter_trade(df)

st.markdown(f"### 💡 Trade Signal: `{signal}`" if should_trade else "### ✅ No Trade Signal")

# Display trade decision
st.write("Latest price:", round(latest_price, 2))
st.write("Trade decision:", "Enter trade" if should_trade else "Hold")

# Refresh every 60s
st.markdown("---")
st.write("⏱️ Auto-refresh every 60 seconds")
st_autorefresh = st.empty()
time.sleep(60)
st.experimental_rerun()
