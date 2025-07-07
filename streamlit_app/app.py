import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time, io, pickle, requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import streamlit as st
import pytz
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload, MediaFileUpload
import ccxt

FEATURES = [
    'EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal',
    'ATR', 'ROC', 'OBV', 'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP'
]

MODEL_FILE = "model.pkl"
DATA_FILE = "btc_data.csv"
LAST_TRAIN_FILE = "last_train.txt"
RETRAIN_INTERVAL = timedelta(hours=12)
FOLDER_NAME = "StreamlitITB"

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]
creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

st.set_page_config(layout='wide')
st.title("🤖 BTC AI Dashboard")

# ======= THRESHOLDS =======
long_thresh = 0.6
short_thresh = 0.6

mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
est = pytz.timezone('US/Eastern')
exchange = ccxt.coinbase()
logfile = "btc_alert_log.csv"
if not os.path.exists(logfile):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(logfile, index=False)

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

def fetch_paginated_ohlcv(symbol='BTC/USDT', timeframe='30m', limit=1000):
    exchange = ccxt.coinbase()
    all_data = []
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    all_data.extend(data)
    df = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

def load_or_fetch_data():
    trained_flag = "trained_once.flag"
    def trained_before():
        return os.path.exists(trained_flag) or download_from_drive(trained_flag)
    if not trained_before():
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
        df.to_csv(DATA_FILE, index=False)
        upload_to_drive(DATA_FILE)
        with open(trained_flag, 'w') as f:
            f.write("trained")
        upload_to_drive(trained_flag)
        return df
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            return df
        else:
            st.error("❌ 'Timestamp' column missing in local CSV.")
            return pd.DataFrame()
    if not download_from_drive(DATA_FILE):
        df = fetch_paginated_ohlcv()
        df.reset_index(inplace=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df.to_csv(DATA_FILE, index=False)
        upload_to_drive(DATA_FILE)
        return df
    df = pd.read_csv(DATA_FILE)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

push_user_key = st.secrets["pushover"]["user"]
push_app_token = st.secrets["pushover"]["token"]
def send_push_notification(msg):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": push_app_token,
        "user": push_user_key,
        "message": msg
    })

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

def train_model():
    st.subheader("📚 Training Model")
    progress = st.progress(0, text="Starting...")
    RAW_ROWS = 1000
    raw_df = load_or_fetch_data()
    st.write("Raw rows loaded from source:", len(raw_df))
    df = raw_df.tail(RAW_ROWS)
    st.write("Rows after tail(RAW_ROWS):", len(df))
    df.to_csv(DATA_FILE, index=False)
    upload_to_drive(DATA_FILE)
    progress.progress(10, text="🔒 Backed up raw data to Drive...")
    progress.progress(20, text="🧠 Calculating technical indicators...")
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    st.write("Rows before dropna:", len(df))
    progress.progress(55, text="🎯 Generating labels...")
    # --------- HERE IS THE CHANGED THRESHOLD ---------
    df['Target'] = ((df['Close'].shift(-3) - df['Close']) / df['Close']).apply(
        lambda x: 2 if x > 0.00016 else (0 if x < -0.00016 else 1)
    )
    df.dropna(subset=FEATURES + ['Target'], inplace=True)
    st.write("Rows after dropna (final training set):", len(df))
    X = df[FEATURES]
    y = df['Target']
    st.write("📊 Target class distribution:", y.value_counts(normalize=True))
    expected_classes = [0, 1, 2]
    actual_classes = sorted(y.unique())
    missing_classes = set(expected_classes) - set(actual_classes)
    if missing_classes:
        st.warning(f"⚠️ Missing classes in training data: {missing_classes}")
        return None, None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    progress.progress(65, text="📦 Scaling features and computing class weights...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    class_weights = compute_class_weight('balanced', classes=np.array(expected_classes), y=y_train)
    progress.progress(85, text="🔧 Training RandomForestClassifier (strict original params)...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train)
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
    try:
        importances = model.feature_importances_
        st.subheader("🔑 RandomForest Feature Importances")
        st.bar_chart(pd.Series(importances, index=FEATURES).sort_values(ascending=False))
    except Exception:
        pass
    progress.progress(95, text="💾 Saving model + scaler to Drive...")
    model_bytes = pickle.dumps((model, scaler))
    upload_to_drive_stream(io.BytesIO(model_bytes), MODEL_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    upload_to_drive_content(LAST_TRAIN_FILE, timestamp)
    progress.progress(100, text="✅ Training complete!")
    st.success("🎉 Model trained and uploaded!")
    return model, scaler

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

def get_data():
    df = pd.DataFrame(
        exchange.fetch_ohlcv('BTC/USDT', '30m', limit=1000),
        columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    est = pytz.timezone('US/Eastern')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
    df.set_index('Timestamp', inplace=True)
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    df.dropna(subset=FEATURES, inplace=True)
    X = df[FEATURES]
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
    except Exception as e:
        st.error(f"Error applying model: {e}")
        df['S0'], df['S1'], df['S2'], df['Prediction'] = None, None, None, None
    return df

if mode == "Live":
    st.header("🟢 Live Mode")
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=900000, limit=None, key="live_refresh")
    model = st.session_state.model
    scaler = st.session_state.scaler
    df = get_data()
    features = FEATURES
    X = scaler.transform(df[features])
    raw_probs = model.predict_proba(X)
    full_probs = np.zeros((raw_probs.shape[0], 3))
    for idx, cls in enumerate(model.classes_):
        full_probs[:, cls] = raw_probs[:, idx]
    df['Prediction'] = model.predict(X)
    df['S0'] = full_probs[:, 0]
    df['S1'] = full_probs[:, 1]
    df['S2'] = full_probs[:, 2]
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='lightblue')))
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
    st.subheader("📊 Signal Log (Current Model Predictions)")
    st.dataframe(signal_df, use_container_width=True)
    now_est = datetime.now(est)
    st.write("⏰ Last refreshed:", now_est.strftime("%H:%M:%S"))
    st.markdown("---")
    if st.button("🔁 Force Retrain", type="primary"):
        with st.spinner("Retraining model..."):
            model, scaler = train_model()
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success("✅ Model retrained successfully.")
            st.rerun()

elif mode == "Backtest":
    df = get_data()
    trades = []
    in_position = None
    entry_time = entry_price = entry_row = None
    for i in range(1, len(df)):
        row = df.iloc[i]
        if in_position is None:
            valid_long = row['Prediction'] == 2 and row['S2'] > long_thresh
            valid_short = row['Prediction'] == 0 and row['S0'] > short_thresh
            if valid_long:
                in_position = "LONG"
                entry_time, entry_price, entry_row = row.name, row['Close'], row
            elif valid_short:
                in_position = "SHORT"
                entry_time, entry_price, entry_row = row.name, row['Close'], row
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
                    "Confidence": round(max(entry_row.get("S0", 0), entry_row.get("S2", 0)), 3)
                })
                in_position = None
    df_trades = pd.DataFrame(trades)
    st.subheader("🧪 Backtest — Strict Model")
    if not df_trades.empty:
        st.dataframe(df_trades.style.applymap(lambda v: 'color:green' if v > 0 else 'color:red', subset=["PNL (USD)", "Profit %"]))
        st.markdown(f"**Total Trades**: {len(df_trades)} | **Avg Profit**: {df_trades['Profit %'].mean():.2f}%")
    else:
        st.warning("No trades in this backtest.")
    st.subheader("📈 Backtest Chart with Trades")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='lightblue')))
    color_map = {"LONG": "green", "SHORT": "red"}
    for _, trade in df_trades.iterrows():
        color = color_map.get(trade["Direction"], "gray")
        fig.add_trace(go.Scatter(
            x=[trade["Entry Time"]], y=[trade["Entry Price"]],
            mode='markers',
            marker=dict(color=color, symbol='circle', size=10),
            name=f"{trade['Direction']} Entry"
        ))
        fig.add_trace(go.Scatter(
            x=[trade["Exit Time"]], y=[trade["Exit Price"]],
            mode='markers',
            marker=dict(color=color, symbol='x', size=10),
            name=f"{trade['Direction']} Exit"
        ))
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
