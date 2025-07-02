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
    existing = drive_service.files().list(q=f"name='{filename}' and '{folder_id}' in parents", fields='files(id)').execute().get('files', [])
    if existing:
        drive_service.files().delete(fileId=existing[0]['id']).execute()
    drive_service.files().create(body=file_metadata, media_body=media).execute()

def upload_to_drive_content(filename, content):
    with open("temp.txt", "w") as f:
        f.write(content)
    upload_to_drive("temp.txt")

def upload_to_drive(filename):
    folder_id = get_folder_id()
    media = MediaFileUpload(filename, resumable=True)
    file_metadata = {'name': filename, 'parents': [folder_id]}
    existing = drive_service.files().list(q=f"name='{filename}' and '{folder_id}' in parents", fields='files(id)').execute().get('files', [])
    if existing:
        drive_service.files().delete(fileId=existing[0]['id']).execute()
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
    exchange = ccxt.coinbase()
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '1m', limit=1500),
                      columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

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

    df['Target'] = ((df['Close'].shift(-3) - df['Close']) / df['Close']).apply(
        lambda x: 2 if x > 0.0015 else (0 if x < -0.0015 else 1))
    df.dropna(inplace=True)

    target_counts = df['Target'].value_counts()
    st.write("### 📊 Class Distribution in Training Data")
    st.bar_chart(target_counts)
    if len(target_counts) < 3:
        st.warning("⚠️ Not all classes present in training data. Consider increasing limit or adjusting thresholds.")

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV',
                'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP']
    X = df[features]
    y = df['Target']
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=50).fit(scaler.transform(X), y)

    model_bytes = pickle.dumps((model, scaler))
    upload_to_drive_stream(io.BytesIO(model_bytes), MODEL_FILE)
    upload_to_drive_content(LAST_TRAIN_FILE, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.success("✅ Model and timestamp uploaded to Drive.")

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
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '1m', limit=750),
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
    df = get_data()
    row = df.iloc[-1]
    price = row['Close']
    pred = row['Prediction']

    s0 = row.get('S0')
    s2 = row.get('S2')
    if s0 is not None and s2 is not None:
        conf = max(s0, s2)
    else:
        conf = 0
        st.warning("⚠️ Confidence score missing — setting to 0.")

    t = row.name.strftime("%Y-%m-%d %H:%M")
    decision = row['ITB']

    if 'open_trade' not in st.session_state:
        st.session_state['open_trade'] = None

    trade = st.session_state['open_trade']

    if not trade and decision in ["LONG", "SHORT"]:
        signal_code = 2 if decision == "LONG" else 0
        st.session_state['open_trade'] = {
            "signal": signal_code,
            "entry_price": price,
            "entry_time": t,
            "entry_conf": conf
        }
        msg = f"BTC 📥 ENTRY — {decision} | {t} | ${price:.2f} | Conf: {conf:.2f}"
        send_push_notification(msg)
        pd.DataFrame([{"Timestamp": t, "Price": price, "Signal": f"ENTRY {decision}", "Scores": f"{conf:.2f}"}]).to_csv(logfile, mode='a', header=False, index=False)

    elif trade:
        reason = None
        if pred != trade["signal"]:
            reason = "Signal flipped"
        elif conf < 0.6:
            reason = "Confidence dropped"

        if reason:
            name = "LONG" if trade["signal"] == 2 else "SHORT"
            msg = f"BTC ❌ EXIT — {name} | {t} | ${price:.2f} | Reason: {reason}"
            send_push_notification(msg)
            pd.DataFrame([{"Timestamp": t, "Price": price, "Signal": f"EXIT {name}", "Scores": reason}]).to_csv(logfile, mode='a', header=False, index=False)
            st.session_state['open_trade'] = None

    st.subheader(f"📊 BTC Live — ${price:.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name="EMA9", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name="EMA21", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name="VWAP", line=dict(color='red')))

    df_long = df[(df['Prediction'] == 2) & (df['S2'] > 0.6)]
    df_short = df[(df['Prediction'] == 0) & (df['S0'] > 0.6)]

    fig.add_trace(go.Scatter(x=df_long.index, y=df_long['Close'], mode='markers', name='📈 Long', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=df_short.index, y=df_short['Close'], mode='markers', name='📉 Short', marker=dict(color='red')))

    fig.update_layout(title=f"📈 BTC {mode} — ${price:.2f}",
                      xaxis_title='Time', yaxis_title='Price',
                      legend=dict(x=1, y=1),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'))

    st.plotly_chart(fig, use_container_width=True)

    # ✅ Force Retrain Button Under Chart
    with st.container():
        st.markdown("---")
        if st.button("🔁 Force Retrain", type="primary"):
            with st.spinner("Retraining model..."):
                model, scaler = train_model()
                st.success("✅ Model retrained successfully.")
                st.rerun()

    st.dataframe(pd.read_csv(logfile).tail(10))
