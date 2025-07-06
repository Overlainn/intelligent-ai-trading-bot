# --- Your existing imports ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Core libraries
import time, io, pickle, requests
from datetime import datetime, date, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# Data and ML
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ======= FEATURE SET ========
FEATURES = [
    'EMA9_Cross_21',
    'EMA12_Cross_26',
    'Above_VWAP',
    'RSI',
    'ADX',
    'MACD',
    'ATR',
    'OBV',
    'Return_1',    # 1-bar return
    'Return_3',    # 3-bar return
    'BB_Width',    # Bollinger Band width
    'Above_20SMA', # Regime: above/below 20SMA
    'Above_50SMA', # Regime: above/below 50SMA
    'Volume_Spike', # 1 if volume > 1.5x 20-bar avg
    'HourOfDay',   # Hour of day (int)
    'DayOfWeek',   # Day of week (int)
]

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

long_thresh = 0.50
short_thresh = 0.50

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

# ========== UI ==========
st.set_page_config(layout='wide')
st.title("🤖 BTC AI Dashboard + ITB Strategy")
# --- Add this: ---
st.sidebar.header("Signal Probability Thresholds")
long_thresh = st.sidebar.slider(
    'Long signal probability threshold', min_value=0.3, max_value=0.95, value=0.60, step=0.01
)
short_thresh = st.sidebar.slider(
    'Short signal probability threshold', min_value=0.3, max_value=0.95, value=0.60, step=0.01
)
mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
est = pytz.timezone('US/Eastern')
exchange = ccxt.coinbase()
logfile = "btc_alert_log.csv"
if not os.path.exists(logfile):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(logfile, index=False)

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

def train_model():
    st.subheader("📚 Training Model")
    progress = st.progress(0, text="Starting...")

    # Step 1: Load Data
    EXTRA = 60
    RAW_ROWS = 44000 + EXTRA
    raw_df = load_or_fetch_data()
    st.write("Raw rows loaded from source:", len(raw_df))
    df = raw_df.tail(RAW_ROWS)
    st.write("Rows after tail(RAW_ROWS):", len(df))

    # Step 2: Save and sync raw file
    df.to_csv(DATA_FILE, index=False)
    upload_to_drive(DATA_FILE)
    progress.progress(10, text="🔒 Backed up raw data to Drive...")

    # Step 3: Feature Engineering
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
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    st.write("Rows after feature engineering:", len(df))

    # Step 3b: Binary/cross features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)

    # Step 3c: Advanced/new features
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_3'] = df['Close'].pct_change(3)
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Close'])
    df['Above_20SMA'] = (df['Close'] > df['Close'].rolling(20).mean()).astype(int)
    df['Above_50SMA'] = (df['Close'] > df['Close'].rolling(50).mean()).astype(int)
    df['Volume_Spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
    df['HourOfDay'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    st.write("Rows before dropna:", len(df))

    # Step 4: Target Engineering (ATR-based thresholds)
    progress.progress(55, text="🎯 Generating labels...")
    future_return = (df['Close'].shift(-4) - df['Close']) / df['Close']
    atr_threshold = 0.2 * df['ATR'] / df['Close']
    static_threshold = 0.0015
    threshold = np.maximum(atr_threshold, static_threshold)
    df['Target'] = np.where(future_return > threshold, 2,
                            np.where(future_return < -threshold, 0, 1))

    # Drop rows with NaNs in features or target
    df.dropna(subset=FEATURES + ['Target'], inplace=True)
    st.write("Rows after dropna (final training set):", len(df))

    # Step 5: Prepare training set
    X = df[FEATURES]
    y = df['Target']
    st.write("📊 Target class distribution:", y.value_counts(normalize=True))

    expected_classes = [0, 1, 2]
    actual_classes = sorted(y.unique())
    missing_classes = set(expected_classes) - set(actual_classes)
    if missing_classes:
        st.warning(f"⚠️ Missing classes in training data: {missing_classes}")
        return None, None

    # --- Split into train/validation ---
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    progress.progress(65, text="📦 Scaling features and computing class weights...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    class_weights = compute_class_weight('balanced', classes=np.array(expected_classes), y=y_train)

    # Step 6: Train final model with your best params
    progress.progress(85, text="🔧 Training XGBoost model (best params)...")
    best_params = {
        "n_estimators": 186,
        "max_depth": 8,
        "learning_rate": 0.0005349070362427248,
        "subsample": 0.5869836653397046,
        "colsample_bytree": 0.65743097026186,
        "reg_lambda": 0.04368036990918477,
        "reg_alpha": 0.0014410899603732284,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
    }
    model = XGBClassifier(**best_params)
    model.fit(X_train_scaled, y_train)

    # --- Model Diagnostics ---
    st.subheader("🔍 Model Diagnostics")
    y_pred = model.predict(X_val_scaled)
    present_labels = sorted([l for l in [0, 1, 2] if l in np.unique(np.concatenate([y_val, y_pred]))])
    all_names = ["Short", "Neutral", "Long"]
    present_names = [all_names[i] for i in present_labels]

    if set([0, 1, 2]) - set(np.unique(y_val)):
        st.warning(f"⚠️ Validation set is missing these classes: {set([0,1,2])-set(np.unique(y_val))}")

    if len(present_labels) < 2:
        st.warning("⚠️ Not enough classes in validation set for diagnostics (need at least 2). "
                   "Try increasing data window or adjusting thresholds.")
    else:
        report = classification_report(
            y_val, y_pred, labels=present_labels, target_names=present_names, zero_division=0
        )
        st.code(report, language='text')

        cm = confusion_matrix(y_val, y_pred, labels=present_labels)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=present_names, yticklabels=present_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    # Feature importances
    importances = model.feature_importances_
    st.subheader("🔑 XGBoost Feature Importances")
    st.bar_chart(pd.Series(importances, index=FEATURES).sort_values(ascending=False))

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

features = FEATURES

# ========== Model/Scaler session state initialization ==========

def get_fresh_model():
    if should_retrain():
        model, scaler = train_model()
        save_last_train_time()
    else:
        model, scaler = load_model_from_drive()
    return model, scaler

if "model" not in st.session_state or "scaler" not in st.session_state:
    st.session_state.model, st.session_state.scaler = get_fresh_model()

model = st.session_state.model
scaler = st.session_state.scaler

# ========== Data Function ==========
def get_data():
    df = pd.DataFrame(
        exchange.fetch_ohlcv('BTC/USDT', '15m', limit=5000),
        columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    est = pytz.timezone('US/Eastern')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
    df.set_index('Timestamp', inplace=True)

    # === Feature Engineering: MUST match train_model() exactly! ===
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

    # Add cross/binary features
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)

    # New engineered features
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_3'] = df['Close'].pct_change(3)
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Close'])
    df['Above_20SMA'] = (df['Close'] > df['Close'].rolling(20).mean()).astype(int)
    df['Above_50SMA'] = (df['Close'] > df['Close'].rolling(50).mean()).astype(int)
    df['Volume_Spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
    df['HourOfDay'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    # Use only these features for modeling
    features = FEATURES
    df.dropna(subset=features, inplace=True)  # drop if any of these features are NaN
    X = df[features]

    try:
        transformed = scaler.transform(X)
        probs = model.predict_proba(transformed)
        padded_probs = [[None, None, None] for _ in range(len(X))]

        for i, row in enumerate(probs):
            for j, val in enumerate(row):
                if j < 3:
                    padded_probs[i][j] = val

        df = df.loc[X.index]
        df[['S0', 'S1', 'S2']] = padded_probs
        df['Prediction'] = model.predict(transformed)

        current_position = st.session_state.get('open_trade', None)
        df['ITB'] = df.apply(lambda row: should_enter_trade(row, current_position), axis=1)

    except Exception as e:
        st.error(f"Error applying model or ITB logic: {e}")
        df['S0'], df['S1'], df['S2'], df['Prediction'], df['ITB'] = None, None, None, None, None

    return df

# ========== Live Mode ==========
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
    features = FEATURES
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
        if row['Prediction'] == 2 and row['S2'] > long_thresh:
            signal = 'Long'
            confidence = row['S2']
        elif row['Prediction'] == 0 and row['S0'] > short_thresh:
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

    # ✅ Plot current signals (from this model run)
    long_signals = df[(df['Prediction'] == 2) & (df['S2'] > long_thresh)]
    short_signals = df[(df['Prediction'] == 0) & (df['S0'] > short_thresh)]

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
                valid_long = row['Prediction'] == 2 and row['S2'] > long_thresh
                valid_short = row['Prediction'] == 0 and row['S0'] > short_thresh
                if valid_long and passes_itb:
                    in_position = "LONG"
                    entry_time, entry_price, entry_row = row.name, row['Close'], row
                elif valid_short and passes_itb:
                    in_position = "SHORT"
                    entry_time, entry_price, entry_row = row.name, row['Close'], row

            # === EXIT LOGIC ===
            elif in_position == "LONG":
                valid_exit = row['Prediction'] == 0 and row['S0'] > short_thresh
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
                valid_exit = row['Prediction'] == 2 and row['S2'] > long_thresh
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
    signal_longs = df[(df['Prediction'] == 2) & (df['S2'] > long_thresh)]
    signal_shorts = df[(df['Prediction'] == 0) & (df['S0'] > short_thresh)]

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
