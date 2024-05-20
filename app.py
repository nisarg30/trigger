from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import joblib
import io
from keras.models import load_model
import tensorflow as tf
import os
import logging

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
logger.info("Loading models...")
nifty_1_buy_bilstm = load_model('./models/nifty_models_1min/NIFTY_1_bilstm_model.keras')
nifty_1_buy_lstm = load_model('./models/nifty_models_1min/NIFTY_1_lstm_model.keras')
nifty_1_buy_log = joblib.load('./models/nifty_models_1min/NIFTY_1_log_model.joblib')

nifty_1_sell_bilstm = load_model('./models/nifty_models_1min_sell/NIFTY_1_bilstm_sell_model.keras')
nifty_1_sell_lstm = load_model('./models/nifty_models_1min_sell/NIFTY_1_lstm_sell_model.keras')
nifty_1_sell_log = joblib.load('./models/nifty_models_1min_sell/NIFTY_1_log_sell_model.joblib')

nifty_5_buy_bilstm = load_model('./models/nifty_models_5min/NIFTY_5_bilstm_model.keras')
nifty_5_buy_lstm = load_model('./models/nifty_models_5min/NIFTY_5_lstm_model.keras')
nifty_5_buy_log = joblib.load('./models/nifty_models_5min/NIFTY_5_log_model.joblib')

nifty_5_sell_bilstm = load_model('./models/nifty_models_5min_sell/NIFTY_5_bilstm_sell_model.keras')
nifty_5_sell_lstm = load_model('./models/nifty_models_5min_sell/NIFTY_5_lstm_sell_model.keras')
nifty_5_sell_log = joblib.load('./models/nifty_models_5min_sell/NIFTY_5_log_sell_model.joblib')

nifty1_1_buy_bilstm = load_model('./models/nifty1_models_1min/NIFTY1!_1_bilstm_model.keras')
nifty1_1_buy_lstm = load_model('./models/nifty1_models_1min/NIFTY1!_1_lstm_model.keras')
nifty1_1_buy_log = joblib.load('./models/nifty1_models_1min/NIFTY1!_1_log_model.joblib')

nifty1_1_sell_bilstm = load_model('./models/nifty1_models_1min_sell/NIFTY1!_1_bilstm_sell_model.keras')
nifty1_1_sell_lstm = load_model('./models/nifty1_models_1min_sell/NIFTY1!_1_lstm_sell_model.keras')
nifty1_1_sell_log = joblib.load('./models/nifty1_models_1min_sell/NIFTY1!_1_log_sell_model.joblib')

nifty1_5_buy_bilstm = load_model('./models/nifty1_models_5min/NIFTY1!_5_bilstm_model.keras')
nifty1_5_buy_lstm = load_model('./models/nifty1_models_5min/NIFTY1!_5_lstm_model.keras')
nifty1_5_buy_log = joblib.load('./models/nifty1_models_5min/NIFTY1!_5_log_model.joblib')

nifty1_5_sell_bilstm = load_model('./models/nifty1_models_5min_sell/NIFTY1!_5_bilstm_sell_model.keras')
nifty1_5_sell_lstm = load_model('./models/nifty1_models_5min_sell/NIFTY1!_5_lstm_sell_model.keras')
nifty1_5_sell_log = joblib.load('./models/nifty1_models_5min_sell/NIFTY1!_5_log_sell_model.joblib')

logger.info("Loading models completed...")

def add_technical_indicators(df):
    logger.info("Adding technical indicators...")
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    df = df[::-1]   # Reverse the DataFrame
    
    df.loc[:, 'ema_20'] = ta.ema(df['Close'], length=20)
    df.loc[:, 'ema_50'] = ta.ema(df['Close'], length=50)
    df.loc[:, 'rsi'] = ta.rsi(df['Close'], length=14)
    df.loc[:, 'macd'] = ta.macd(df['Close'])['MACD_12_26_9']
    df.loc[:, 'signal_line'] = ta.macd(df['Close'])['MACDs_12_26_9']
    
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df.loc[:, 'upper_band'] = bbands['BBU_20_2.0']
    df.loc[:, 'middle_band'] = bbands['BBM_20_2.0']
    df.loc[:, 'lower_band'] = bbands['BBL_20_2.0']
    
    df = df[::-1]  # Reverse the DataFrame back to its original order
    df = df.dropna()  # Drop rows with NaN values
    logger.debug(f"Technical indicators added. Data shape: {df.shape}")
    return df

def calculate_new_columns(df):
    logger.info("Calculating new columns...")
    df = df.copy()
    df['wvff'] = df['wvf'] - df['rangeHigh']
    df['ema'] = df['ema_20'] - df['ema_50']
    df['cema'] = df['Close'] - df['ema_20']
    df['lowc'] = df['Close'] - df['lower_band']
    df['macdf'] = df['macd'] - df['signal_line']
    df['midlc'] = df['Close'] - df['middle_band']
    df['wvflf'] = df['wvfl'] - df['rangeHighl']
    df = df.drop(columns=['Timestamp', 'Volume', 'Open', 'Max', 'Min', 'wvf', 'rangeHigh', 'wvfl', 'rangeHighl', 'ema_20', 'ema_50', 'macd', 'signal_line', 'upper_band', 'lower_band', 'middle_band'])
    logger.debug(f"New columns calculated. Data shape: {df.shape}")
    return df

def calculate_diff_columns(data):
    logger.info("Calculating difference columns...")
    xx = data.copy()
    xx = xx[::-1]
    df_diff = xx.diff().fillna(0)
    df_diff = df_diff[::-1]
    logger.debug(f"Difference columns calculated. Data shape: {df_diff.shape}")
    return df_diff

def scaling(df):
    logger.info("Scaling data...")
    features = df.columns.tolist()
    scaler = MinMaxScaler(feature_range=(-100, 100))
    first_61_rows = df.iloc[:61]
    sequence_scaled = first_61_rows[features].apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0).values.tolist()
    sequence_reversed = sequence_scaled[::-1]
    logger.debug(f"Data scaling completed. Data shape: {len(sequence_reversed)} sequences")
    return sequence_reversed

def prepare_data(data):
    logger.info("Preparing data...")
    data_with_indicators = add_technical_indicators(data)
    data_processed = calculate_new_columns(data_with_indicators)
    data_diff = calculate_diff_columns(data_processed.copy())
    data_diff = scaling(data_diff.copy())
    data_processed = scaling(data_processed.copy())
    X1 = []
    X1.append(data_processed)
    X2 = []
    X2.append(data_diff)
    logger.debug(f"Data prepared. Data shapes: {np.array(X1).shape}, {np.array(X2).shape}")
    return X1, X2

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/nifty_1', methods=['POST'])
def nifty_1():
    csv_data = request.get_json().get('csvData')
    direction = request.get_json().get('dir')
    logger.info("Received request for /nifty_1 endpoint.")
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)
    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    result = False
    if direction == 1:
        logger.info("Processing buy direction...")
        xxc = nifty_1_buy_bilstm.predict(data_without_diff)
        xxl = nifty_1_buy_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty_1_buy_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        logger.info("Processing sell direction...")
        xxc = nifty_1_sell_bilstm.predict(data_without_diff)
        xxl = nifty_1_sell_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty_1_sell_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    logger.debug(f"Prediction result: {result}")
    return jsonify(result.tolist())

@app.route('/nifty_5', methods=['POST'])
def nifty_5():
    csv_data = request.get_json().get('csvData')
    direction = request.get_json().get('dir')
    logger.info("Received request for /nifty_5 endpoint.")
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)
    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    result = False
    if direction == 1:
        logger.info("Processing buy direction...")
        xxc = nifty_5_buy_bilstm.predict(data_without_diff)
        xxl = nifty_5_buy_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty_5_buy_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        logger.info("Processing sell direction...")
        xxc = nifty_5_sell_bilstm.predict(data_without_diff)
        xxl = nifty_5_sell_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty_5_sell_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    logger.debug(f"Prediction result: {result}")
    return jsonify(result.tolist())

@app.route('/nifty1_1', methods=['POST'])
def nifty1_1():
    csv_data = request.get_json().get('csvData')
    direction = request.get_json().get('dir')
    logger.info("Received request for /nifty1_1 endpoint.")
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)
    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    result = False
    if direction == 1:
        logger.info("Processing buy direction...")
        xxc = nifty1_1_buy_bilstm.predict(data_without_diff)
        xxl = nifty1_1_buy_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty1_1_buy_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        logger.info("Processing sell direction...")
        xxc = nifty1_1_sell_bilstm.predict(data_without_diff)
        xxl = nifty1_1_sell_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty1_1_sell_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    logger.debug(f"Prediction result: {result}")
    return jsonify(result.tolist())

@app.route('/nifty1_5', methods=['POST'])
def nifty1_5():
    csv_data = request.get_json().get('csvData')
    direction = request.get_json().get('dir')
    logger.info("Received request for /nifty1_5 endpoint.")
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)
    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    result = False
    if direction == 1:
        logger.info("Processing buy direction...")
        xxc = nifty1_5_buy_bilstm.predict(data_without_diff)
        xxl = nifty1_5_buy_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty1_5_buy_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        logger.info("Processing sell direction...")
        xxc = nifty1_5_sell_bilstm.predict(data_without_diff)
        xxl = nifty1_5_sell_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty1_5_sell_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    logger.debug(f"Prediction result: {result}")
    return jsonify(result.tolist())

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8080))
#     app.run(host='0.0.0.0', port=port)
