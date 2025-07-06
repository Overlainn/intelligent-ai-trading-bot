# --- Your existing imports ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Core libraries
import time, io, pickle, requests
from datetime import datetime, date, timedelta
from streamlit_autorefresh import st_autorefresh

# Data and ML
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier

SHARED_FEATURES = [
    'EMA12_Cross_26',
    'Above_VWAP',
    'RSI',
    'MACD',
    'ATR',
    'OBV']
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Visualization
import plotly.graph_objs as go

# Web app and timezone
import streamlit as st
import pytz

# Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload, MediaFileUpload

# Trading logic
import ccxt
from trading_engine.strategy import should_enter_trade

# ✅ Model/scaler + Training + Auto‑load logic
import joblib

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "btc_data.csv"
LAST_TRAIN_FILE = "last_train.txt"
RETRAIN_INTERVAL = timedelta(hours=12)
FOLDER_NAME = "StreamlitITB"

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]
creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

# ========== Google Drive Functions ==========
def get_folder_id():
    query = f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
    response = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    files = response.get('files', [])
    if files:
        return files[0]['id']
    file_metadata = {'name': FOLDER_NAME, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = drive_service.files().create(body=file_metadata, fields='id').execute()
    return folder['id']

def upload_to_drive(filename, drive_name=None):
    folder_id = get_folder_id()
    file_name = drive_name or os.path.basename(filename)
    media = MediaFileUpload(filename, resumable=True)
    body = {'name': file_name, 'parents': [folder_id]}

    existing = drive_service.files().list(
        q=f"name='{file_name}' and '{folder_id}' in parents",
        fields='files(id)'
    ).execute().get('files', [])

    if existing:
        drive_service.files().delete(fileId=existing[0]['id']).execute()
    drive_service.files().create(body=body, media_body=media).execute()

def upload_to_drive_stream(file_stream, filename):
    folder_id = get_folder_id()
    media = MediaIoBaseUpload(file_stream, mimetype='application/octet-stream', resumable=True)
    file_metadata = {'name': filename, 'parents': [folder_id]}

    existing = drive_service.files().list(
        q=f"name='{filename}' and '{folder_id}' in parents",
        fields='files(id)'
    ).execute().get('files', [])

    if existing:
        try:
            drive_service.files().delete(fileId=existing[0]['id']).execute()
        except Exception as e:
            print(f"⚠️ Warning: Couldn't delete existing file '{filename}': {e}")
    drive_service.files().create(body=file_metadata, media_body=media).execute()

def upload_to_drive_content(filename, content):
    with open(filename, "w") as f:
        f.write(content)
    upload_to_drive(filename)

def download_from_drive(filename):
    folder_id = get_folder_id()
    results = drive_service.files().list(q=f"name='{filename}' and '{folder_id}' in parents", fields="files(id)").execute()
    files = results.get('files', [])
    if not files:
        return False
    file_id = files[0]['id']
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    with open(filename, 'wb') as f:
        f.write(fh.getvalue())
    return True

# ========== Historical Data Fetching ==========
def fetch_paginated_ohlcv(symbol='BTC/USDT', timeframe='15m', days=90):
    exchange = ccxt.coinbase()
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_data = []
    while True:
        data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not data:
            break
        all_data.extend(data)
        since = data[-1][0] + 60_000
        if len(data) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df


def load_or_fetch_data():
    trained_flag = "trained_once.flag"

    # Check if we already trained on BTC_4_Candle.csv
    def trained_before():
        return os.path.exists(trained_flag) or download_from_drive(trained_flag)

    if not trained_before():
        # 🔁 First-time training from BTC_4_Candle.csv
        if not download_from_drive("BTC_4_Candle.csv"):
            st.error("❌ Could not find BTC_4_Candle.csv on Google Drive.")
            return pd.DataFrame()
        try:
            df = pd.read_csv("BTC_4_Candle.csv")
            if "Timestamp" not in df.columns:
                st.error("❌ 'Timestamp' column missing in BTC_4_Candle.csv.")
                return pd.DataFrame()
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        except Exception as e:
            st.error(f"❌ Failed to read BTC_4_Candle.csv: {e}")
            return pd.DataFrame()

        # ✅ Save local copy to keep consistent with the rest of app
        df.to_csv(DATA_FILE, index=False)
        upload_to_drive(DATA_FILE)

        # ✅ Create flag file so we never use BTC_4_Candle.csv again
        with open(trained_flag, 'w') as f:
            f.write("trained")
        upload_to_drive(trained_flag)

        return df

    # ✅ From now on: use local btc_data.csv or fetch online if missing
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            return df
        else:
            st.error("❌ 'Timestamp' column missing in local CSV.")
            return pd.DataFrame()

    # 🔁 No local data file, try fetching online
    if not download_from_drive(DATA_FILE):
        df = fetch_paginated_ohlcv()
        df.reset_index(inplace=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df.to_csv(DATA_FILE, index=False)
        upload_to_drive(DATA_FILE)
        return df

    # ✅ File downloaded from Drive
    df = pd.read_csv(DATA_FILE)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

# ========== Push Notifications ==========
push_user_key = st.secrets["pushover"]["user"]
push_app_token = st.secrets["pushover"]["token"]
def send_push_notification(msg):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": push_app_token,
        "user": push_user_key,
        "message": msg
    })

# ========== Refresh Timer ==========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========== Model Training ==========
def train_model():
    st.subheader("📚 Training Model")
    progress = st.progress(0, text="Starting...")

    

    # Step 1: Load Data
    progress.progress(5, text="📥 Loading dataset...")
    df = load_or_fetch_data()# Step 1: Load Data
    df = df.tail(50000)  # <-- Only keep the most recent 50,000 rows
    st.write(f"Training on {len(df)} most recent rows.")

    # Step 2: Save and sync raw file
    df.to_csv(DATA_FILE, index=False)
    upload_to_drive(DATA_FILE)
    progress.progress(10, text="🔒 Backed up raw data to Drive...")

    # Step 3: Feature Engineering — only features needed for selected set
    progress.progress(20, text="🧠 Calculating technical indicators...")
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    # Only these will be used as model input:
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    progress.progress(45, text="✅ Features engineered.")

    # Step 4: Target Engineering
    progress.progress(55, text="🎯 Generating labels...")
    df['Target'] = ((df['Close'].shift(-4) - df['Close']) / df['Close']).apply(
        lambda x: 2 if x > 0.0003 else (0 if x < -0.0003 else 1)
    )
    df.dropna(inplace=True)

    # Step 5: Prepare training set — ONLY these features
    features = [
        'EMA12_Cross_26',
        'Above_VWAP',
        'RSI',
        'MACD',
        'ATR',
        'OBV'
    ]
    X = df[features]
    y = df['Target']

    class_counts = y.value_counts(normalize=True)
    st.write("📊 Target class distribution:", class_counts)

    expected_classes = [0, 1, 2]
    actual_classes = sorted(y.unique())
    missing_classes = set(expected_classes) - set(actual_classes)
    if missing_classes:
        st.warning(f"⚠️ Missing classes in training data: {missing_classes}")
        return None, None

    progress.progress(65, text="📦 Scaling features and computing class weights...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_weights = compute_class_weight('balanced', classes=np.array(expected_classes), y=y)
    weight_dict = dict(zip(expected_classes, class_weights))

    # Step 6: Train model
    progress.progress(80, text="🔧 Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_scaled, y)

    # Step 7: Save model + scaler
    progress.progress(95, text="💾 Saving model + scaler to Drive...")
    model_bytes = pickle.dumps((model, scaler))
    upload_to_drive_stream(io.BytesIO(model_bytes), MODEL_FILE)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    upload_to_drive_content(LAST_TRAIN_FILE, timestamp)

    progress.progress(100, text="✅ Training complete!")
    st.success("🎉 Model trained and uploaded!")

    return model, scaler
  
# ========== Utility Functions ==========

def save_last_train_time():
    try:
        with open(LAST_TRAIN_FILE, 'w') as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        upload_to_drive(LAST_TRAIN_FILE)
    except Exception as e:
        st.error(f"❌ Failed to save last train time: {e}")

def load_model_from_drive():
    if not download_from_drive(MODEL_FILE):
        st.error("❌ Failed to load model from Drive. Training new one.")
        return train_model()
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

# ========== Local Training Functions ==========
def load_model_and_scaler():
    """
    Loads the model and scaler from local pickle file if available.
    Retrains from scratch if loading fails.
    """
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model, scaler = pickle.load(f)
            return model, scaler
        except Exception as e:
            st.warning(f"⚠️ Failed to load local model: {e}")
            st.info("🔁 Re-training model...")
    return train_model()

def load_model_from_drive():
    """
    Loads the model+scaler from Google Drive (via pickle).
    If unavailable, trains from scratch.
    """
    if not download_from_drive(MODEL_FILE):
        st.error("❌ Failed to download model from Drive. Training new one.")
        return train_model()

    try:
        with open(MODEL_FILE, 'rb') as f:
            model, scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.warning(f"⚠️ Error reading model file: {e}")
        return train_model()

# ✅ Load model/scaler at startup
model, scaler = load_model_from_drive()

# ========== Model/Scaler session state initialization ==========
if "model" not in st.session_state or "scaler" not in st.session_state:
    model, scaler = load_model_from_drive()
    st.session_state.model = model
    st.session_state.scaler = scaler
else:
    model = st.session_state.model
    scaler = st.session_state.scaler

# ========== Load or Retrain Model ==========
RETRAIN_INTERVAL = timedelta(hours=12)

def should_retrain():
    if not download_from_drive(LAST_TRAIN_FILE):
        st.warning("📄 No last_train.txt found on Drive. Retraining.")
        return True
    try:
        with open(LAST_TRAIN_FILE, 'r') as f:
            last_train_str = f.read().strip()
        last_train_time = datetime.strptime(last_train_str, "%Y-%m-%d %H:%M:%S")
        if datetime.now() - last_train_time > RETRAIN_INTERVAL:
            st.info("🕒 12 hours passed. Retraining model.")
            return True
        else:
            st.success("✅ Model recently trained. Skipping retrain.")
            return False
    except Exception as e:
        st.error(f"⚠️ Error reading last_train.txt: {e}")
        return True

# Only automatic retrain or load from drive — no sidebar button
if should_retrain():
    model, scaler = train_model()
    save_last_train_time()
else:
    model, scaler = load_model_from_drive()

features = SHARED_FEATURES

# ========== UI ==========
st.set_page_config(layout='wide')
st.title("🤖 BTC AI Dashboard + ITB Strategy")
mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
est = pytz.timezone('US/Eastern')
exchange = ccxt.coinbase()
logfile = "btc_alert_log.csv"
if not os.path.exists(logfile):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(logfile, index=False)

# ========== Data Function ==========
def get_data():
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '15m', limit=5000),
                      columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
    df.set_index('Timestamp', inplace=True)

    # Needed for cross/binary features:
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    # Only these binary/categorical features will be used
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)

    # Use only these features for modeling
    features = SHARED_FEATURES
    df.dropna(subset=features, inplace=True)  # drop if any of these features are NaN
    X = df[features]
    X_indexed = X.copy()

    try:
        transformed = scaler.transform(X)
        probs = model.predict_proba(transformed)
        padded_probs = [[None, None, None] for _ in range(len(X))]

        for i, row in enumerate(probs):
            for j, val in enumerate(row):
                if j < 3:
                    padded_probs[i][j] = val

        df = df.loc[X_indexed.index]
        df[['S0', 'S1', 'S2']] = padded_probs
        df['Prediction'] = model.predict(transformed)

        current_position = st.session_state.get('open_trade', None)
        df['ITB'] = df.apply(lambda row: should_enter_trade(row, current_position), axis=1)

    except Exception as e:
        st.error(f"Error applying model or ITB logic: {e}")
        df['S0'], df['S1'], df['S2'], df['Prediction'], df['ITB'] = None, None, None, None, None

    return df

if mode == "Live":
    st.header("🟢 Live Mode")

    # 🔁 Auto-refresh every 15 minutes
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=900000, limit=None, key="live_refresh")

    # ✅ Always use the model/scaler from session state
    model = st.session_state.model
    scaler = st.session_state.scaler

    # ✅ Load latest live data
    df = get_data()
    features = SHARED_FEATURES
    X = scaler.transform(df[features])
    raw_probs = model.predict_proba(X)

    # Ensure 3 class probabilities
    full_probs = np.zeros((raw_probs.shape[0], 3))
    for idx, cls in enumerate(model.classes_):
        full_probs[:, cls] = raw_probs[:, idx]

    df['Prediction'] = model.predict(X)
    df['S0'] = full_probs[:, 0]
    df['S1'] = full_probs[:, 1]
    df['S2'] = full_probs[:, 2]

    # ========== Build Signal Log for All Candles ==========
    signal_log = []
    for idx, row in df.iterrows():
        signal = None
        confidence = 0
        if row['Prediction'] == 2 and row['S2'] > 0.50:
            signal = 'Long'
            confidence = row['S2']
        elif row['Prediction'] == 0 and row['S0'] > 0.50:
            signal = 'Short'
            confidence = row['S0']
        else:
            signal = "None"
        signal_log.append({
            "Timestamp": idx,
            "Price": round(row['Close'], 2),
            "Signal": signal,
            "Short": round(row['S0'], 4) if row['S0'] is not None else None,
            "Neutral": round(row['S1'], 4) if row['S1'] is not None else None,
            "Long": round(row['S2'], 4) if row['S2'] is not None else None,
            "Confidence": round(confidence, 4)
        })

    signal_df = pd.DataFrame(signal_log)
    signal_df["Timestamp"] = pd.to_datetime(signal_df["Timestamp"], errors="coerce")
    signal_df = signal_df.sort_values(by="Timestamp", ascending=False)

    # ========== Plot Price + Indicators ==========
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='red')))

    # ✅ Plot current signals (from this model run)
    long_signals = df[(df['Prediction'] == 2) & (df['S2'] > 0.50)]
    short_signals = df[(df['Prediction'] == 0) & (df['S0'] > 0.50)]

    fig.add_trace(go.Scatter(
        x=long_signals.index,
        y=long_signals['Close'],
        mode='markers',
        marker=dict(color='green', size=8),
        name='Long Signal'
    ))
    fig.add_trace(go.Scatter(
        x=short_signals.index,
        y=short_signals['Close'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Short Signal'
    ))

    fig.update_layout(
        title=f"📉 BTC Live — ${df['Close'].iloc[-1]:.2f}",
        height=600,
        xaxis_title="Time",
        yaxis_title="Price",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📋 Signal Log Table (current predictions only)
    st.subheader("📊 Signal Log (Current Model Predictions)")
    st.dataframe(signal_df, use_container_width=True)

    # Optionally filter for only actionable signals:
    # actionable = signal_df[signal_df["Signal"].isin(["Long", "Short"])]
    # st.dataframe(actionable, use_container_width=True)

    import pytz
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    st.write("⏰ Last refreshed:", now_est.strftime("%H:%M:%S"))

    # 🔁 Force Retrain button remains as is
    st.markdown("---")
    if st.button("🔁 Force Retrain", type="primary"):
        with st.spinner("Retraining model..."):
            model, scaler = train_model()
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success("✅ Model retrained successfully.")
            st.rerun()

# ========== Backtest Mode ==========
elif mode == "Backtest":
    df = get_data()

    results = {"Pure": [], "ITB": []}

    for use_itb in ["Pure", "ITB"]:
        trades = []
        in_position = None
        entry_time = entry_price = entry_row = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            passes_itb = row.get('ITB', True) if use_itb == "ITB" else True

            # === ENTRY LOGIC ===
            if in_position is None:
                valid_long = row['Prediction'] == 2 and row['S2'] > 0.50
                valid_short = row['Prediction'] == 0 and row['S0'] > 0.50

                if valid_long and passes_itb:
                    in_position = "LONG"
                    entry_time, entry_price, entry_row = row.name, row['Close'], row
                elif valid_short and passes_itb:
                    in_position = "SHORT"
                    entry_time, entry_price, entry_row = row.name, row['Close'], row

            # === EXIT LOGIC ===
            elif in_position == "LONG":
                valid_exit = row['Prediction'] == 0 and row['S0'] > 0.50
                if valid_exit:
                    trades.append({
                        "Entry Time": entry_time,
                        "Exit Time": row.name,
                        "Direction": in_position,
                        "Entry Price": entry_price,
                        "Exit Price": row['Close'],
                        "PNL (USD)": row['Close'] - entry_price,
                        "Profit %": (row['Close'] / entry_price - 1) * 100,
                        "ITB": use_itb,
                        "Confidence": round(max(entry_row.get("S0", 0), entry_row.get("S2", 0)), 3)
                    })
                    in_position = None

            elif in_position == "SHORT":
                valid_exit = row['Prediction'] == 2 and row['S2'] > 0.50
                if valid_exit:
                    trades.append({
                        "Entry Time": entry_time,
                        "Exit Time": row.name,
                        "Direction": in_position,
                        "Entry Price": entry_price,
                        "Exit Price": row['Close'],
                        "PNL (USD)": entry_price - row['Close'],
                        "Profit %": (entry_price / row['Close'] - 1) * 100,
                        "ITB": use_itb,
                        "Confidence": round(max(entry_row.get("S0", 0), entry_row.get("S2", 0)), 3)
                    })
                    in_position = None

        results[use_itb] = trades

    # ✅ Convert to DataFrames
    df_pure = pd.DataFrame(results["Pure"])
    df_itb = pd.DataFrame(results["ITB"])

    # ✅ Display Results
    st.subheader("🧪 Backtest A: Pure Model")
    if not df_pure.empty:
        st.dataframe(df_pure.style.applymap(lambda v: 'color:green' if v > 0 else 'color:red', subset=["PNL (USD)", "Profit %"]))
        st.markdown(f"**Total Trades**: {len(df_pure)} | **Avg Profit**: {df_pure['Profit %'].mean():.2f}%")
    else:
        st.warning("No trades in pure mode.")

    st.subheader("🧪 Backtest B: Model + ITB Filter")
    if not df_itb.empty:
        st.dataframe(df_itb.style.applymap(lambda v: 'color:green' if v > 0 else 'color:red', subset=["PNL (USD)", "Profit %"]))
        st.markdown(f"**Total Trades**: {len(df_itb)} | **Avg Profit**: {df_itb['Profit %'].mean():.2f}%")
    else:
        st.warning("No trades in ITB mode.")

    # ✅ Plot Combined Trades + Model Signals
    st.subheader("📈 Backtest Chart with All Trades")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='lightblue')))

    color_map = {"LONG": "green", "SHORT": "red"}
    for trade_df, label, marker_symbol in [(df_pure, "Pure", 'circle'), (df_itb, "ITB", 'diamond')]:
        for _, trade in trade_df.iterrows():
            color = color_map.get(trade["Direction"], "gray")
            fig.add_trace(go.Scatter(
                x=[trade["Entry Time"]], y=[trade["Entry Price"]],
                mode='markers',
                marker=dict(color=color, symbol=marker_symbol, size=10),
                name=f"{label} {trade['Direction']} Entry"
            ))
            fig.add_trace(go.Scatter(
                x=[trade["Exit Time"]], y=[trade["Exit Price"]],
                mode='markers',
                marker=dict(color=color, symbol='x', size=10),
                name=f"{label} {trade['Direction']} Exit"
            ))

    # ✅ Add Model Signal Markers
    signal_longs = df[(df['Prediction'] == 2) & (df['S2'] > 0.50)]
    signal_shorts = df[(df['Prediction'] == 0) & (df['S0'] > 0.50)]

    fig.add_trace(go.Scatter(
        x=signal_longs.index,
        y=signal_longs['Close'],
        mode='markers',
        marker=dict(color='lime', size=6),
        name='📈 Model Long Signal'
    ))
    fig.add_trace(go.Scatter(
        x=signal_shorts.index,
        y=signal_shorts['Close'],
        mode='markers',
        marker=dict(color='orangered', size=6),
        name='📉 Model Short Signal'
    ))

    fig.update_layout(
        height=600,
        title="Backtest Trade Entries and Exits",
        showlegend=True,
        xaxis_title="Time",
        yaxis_title="Price",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)
