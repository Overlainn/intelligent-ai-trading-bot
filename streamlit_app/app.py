import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ccxt, pandas as pd, ta, time, streamlit as st, plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pytz, requests, pickle, io
from datetime import datetime, date, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload, MediaFileUpload
from trading_engine.strategy import should_enter_trade

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]
creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

MODEL_FILE = "btc_model.pkl"
LAST_TRAIN_FILE = "last_train.txt"
DATA_FILE = "btc_data.csv"
FOLDER_NAME = "StreamlitITB"

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
    with open("last_train.txt", "w") as f:
        f.write(content)
    upload_to_drive("last_train.txt")

def upload_to_drive(filename):
    folder_id = get_folder_id()
    media = MediaFileUpload(filename, resumable=True)
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

def load_model_from_drive():
    if not download_from_drive(MODEL_FILE):
        st.error("❌ Failed to load model from Drive. Training new one.")
        return train_model()
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

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
    if not os.path.exists(DATA_FILE):
        if not download_from_drive(DATA_FILE):
            df = fetch_paginated_ohlcv()
            df.to_csv(DATA_FILE)
            upload_to_drive(DATA_FILE)  # ✅ Only uploaded if freshly fetched
            return df
    return pd.read_csv(DATA_FILE, index_col='Timestamp', parse_dates=True)

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
    df = load_or_fetch_data()

    # 🔒 Ensure data file exists and is up-to-date
    df.to_csv(DATA_FILE)  # Overwrite or create locally
    upload_to_drive(DATA_FILE)  # Upload to Drive

    # Feature Engineering
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)

    df['Future_Close'] = df['Close'].shift(-3)
    df['Pct_Change'] = (df['Future_Close'] - df['Close']) / df['Close']
    df['Target'] = df['Pct_Change'].apply(lambda x: 2 if x > 0.003 else (0 if x < -0.003 else 1))
    df.dropna(inplace=True)

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal',
                'ATR', 'ROC', 'OBV', 'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP']
    X = df[features]
    y = df['Target']

    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=50).fit(scaler.transform(X), y)

    model_bytes = pickle.dumps((model, scaler))
    upload_to_drive_stream(io.BytesIO(model_bytes), MODEL_FILE)
    upload_to_drive_content(LAST_TRAIN_FILE, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return model, scaler

# ========== Load or Retrain Model ==========
RETRAIN_INTERVAL = timedelta(hours=12)

def should_retrain():
    if not download_from_drive(LAST_TRAIN_FILE):
        return True
    with open(LAST_TRAIN_FILE, 'r') as f:
        last_train_str = f.read().strip()
    try:
        last_train_time = datetime.strptime(last_train_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return True
    return datetime.now() - last_train_time > RETRAIN_INTERVAL

# Only automatic retrain or load from drive — no sidebar button
if should_retrain():
    model, scaler = train_model()
else:
    model, scaler = load_model_from_drive()

features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV',
            'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP']

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

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV',
                'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP']

    df.dropna(inplace=True)
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

# ========== Live Mode ==========
if mode == "Live":
    # ✅ Initialize signal log in session state if needed
    if "signal_log" not in st.session_state:
        if download_from_drive("signal_log.csv"):
            try:
                df_loaded = pd.read_csv("signal_log.csv")
                st.session_state.signal_log = df_loaded.to_dict(orient="records")
            except Exception as e:
                st.warning(f"⚠️ Failed to load signal log from Drive: {e}")
                st.session_state.signal_log = []
        else:
            st.session_state.signal_log = []

    # ✅ Get data and make predictions
    df = get_data()
    df['Prediction'] = model.predict(scaler.transform(df[features]))
    probs = model.predict_proba(scaler.transform(df[features]))
    df['S0'] = probs[:, 0]
    df['S1'] = probs[:, 1]
    df['S2'] = probs[:, 2]

    last = df.iloc[-1]
    signal = None
    confidence = 0

    if last['Prediction'] == 2 and last['S2'] > 0.55:
        signal = 'Long'
        confidence = last['S2']
    elif last['Prediction'] == 0 and last['S0'] > 0.55:
        signal = 'Short'
        confidence = last['S0']

    # ✅ Always log the latest row, whether it's a signal or not
    st.session_state.signal_log.append({
        "Timestamp": last.name,
        "Price": round(last['Close'], 2),
        "Signal": signal if signal else "None",
        "Short": round(last['S0'], 4),
        "Neutral": round(last['S1'], 4),
        "Long": round(last['S2'], 4),
        "Confidence": round(confidence, 4)
    })

    # ✅ Keep only last 100 entries
    st.session_state.signal_log = st.session_state.signal_log[-100:]

    # ✅ Save to CSV and upload to Drive
    signal_df = pd.DataFrame(st.session_state.signal_log)
    csv_file = "signal_log.csv"
    signal_df.to_csv(csv_file, index=False)
    upload_to_drive(csv_file)

    # 📈 Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='red')))

    if signal:
        fig.add_trace(go.Scatter(
            x=[last.name], y=[last['Close']],
            mode='markers',
            marker=dict(color='green' if signal == 'Long' else 'red', size=10),
            name=signal
        ))

    fig.update_layout(
        title=f"📉 BTC Live — ${last['Close']:.2f}",
        height=600,
        xaxis_title="Time",
        yaxis_title="Price",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📋 Signal Log Table
    st.subheader("📊 Signal Log")

    # Convert Timestamp to datetime if not already
    signal_df['Timestamp'] = pd.to_datetime(signal_df['Timestamp'], errors='coerce')

    # Sort by latest first
    signal_df_sorted = signal_df.sort_values(by="Timestamp", ascending=False)

    # Display
    st.dataframe(signal_df_sorted)

    # 🔁 Force Retrain Button
    with st.container():
        st.markdown("---")
        if st.button("🔁 Force Retrain", type="primary"):
            with st.spinner("Retraining model..."):
                model, scaler = train_model()
                st.success("✅ Model retrained successfully.")
                st.rerun()

elif mode == "Backtest":
    df = get_data()

    results = {"Pure": [], "ITB": []}

    for use_itb in ["Pure", "ITB"]:
        trades = []
        in_position = None
        entry_time = None
        entry_price = None
        entry_row = None

        for i in range(1, len(df)):
            row = df.iloc[i]

            # Entry logic
            if in_position is None:
                valid_long = row['Prediction'] == 2 and row['S2'] > 0.6
                valid_short = row['Prediction'] == 0 and row['S0'] > 0.6
                passes_itb = row['ITB'] if use_itb == "ITB" else True

                if valid_long and passes_itb:
                    in_position = "LONG"
                    entry_time, entry_price, entry_row = row.name, row['Close'], row
                elif valid_short and passes_itb:
                    in_position = "SHORT"
                    entry_time, entry_price, entry_row = row.name, row['Close'], row

            # Exit logic
            elif in_position == "LONG":
                if row['Prediction'] == 0 and row['S0'] > 0.6:
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
                if row['Prediction'] == 2 and row['S2'] > 0.6:
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

    # 🔎 Convert to DataFrames
    df_pure = pd.DataFrame(results["Pure"])
    df_itb = pd.DataFrame(results["ITB"])

    # 📊 Display Results
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

    # 📈 Plot Combined Trades
    st.subheader("📈 Backtest Chart with All Trades")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='lightblue')))

    for trade_df, mode, symbol in [(df_pure, "Pure", 'circle'), (df_itb, "ITB", 'diamond')]:
        color_map = {'LONG': 'green', 'SHORT': 'red'}
        for _, trade in trade_df.iterrows():
            color = color_map.get(trade["Direction"], 'gray')
            fig.add_trace(go.Scatter(
                x=[trade["Entry Time"]], y=[trade["Entry Price"]],
                mode='markers',
                marker=dict(color=color, symbol=symbol, size=10),
                name=f'{mode} {trade["Direction"]} Entry'
            ))
            fig.add_trace(go.Scatter(
                x=[trade["Exit Time"]], y=[trade["Exit Price"]],
                mode='markers',
                marker=dict(color=color, symbol='x', size=10),
                name=f'{mode} {trade["Direction"]} Exit'
            ))

    fig.update_layout(height=600, title="Backtest Trade Entries and Exits", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
