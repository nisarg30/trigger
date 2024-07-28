from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import joblib
import io
from keras.models import load_model
import tensorflow as tf
import gc

tf.config.run_functions_eagerly(True)

import os
import logging
import psutil

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Function to load models on demand
def load_model_on_demand(model_path):
    return load_model(model_path)

def load_joblib_model_on_demand(model_path):
    return joblib.load(model_path)

# Paths to models
model_paths = {
    'nifty_1_buy_bilstm': './models/nifty_models_1min/NIFTY_1_bilstm_model.keras',
    'nifty_1_buy_lstm': './models/nifty_models_1min/NIFTY_1_lstm_model.keras',
    'nifty_1_buy_log': './models/nifty_models_1min/NIFTY_1_log_model.joblib',
    'banknifty_1_buy_bilstm': './models/banknifty_models_1min/BANKNIFTY_1_bilstm_model.keras',
    'banknifty_1_buy_lstm': './models/banknifty_models_1min/BANKNIFTY_1_lstm_model.keras',
    'banknifty_1_buy_log': './models/banknifty_models_1min/BANKNIFTY_1_log_model.joblib',
    'nifty_1_sell_bilstm': './models/nifty_models_1min_sell/NIFTY_1_bilstm_sell_model.keras',
    'nifty_1_sell_lstm': './models/nifty_models_1min_sell/NIFTY_1_lstm_sell_model.keras',
    'nifty_1_sell_log': './models/nifty_models_1min_sell/NIFTY_1_log_sell_model.joblib',
    'banknifty_1_sell_bilstm': './models/banknifty_models_1min_sell/BANKNIFTY_1_bilstm_sell_model.keras',
    'banknifty_1_sell_lstm': './models/banknifty_models_1min_sell/BANKNIFTY_1_lstm_sell_model.keras',
    'banknifty_1_sell_log': './models/banknifty_models_1min_sell/BANKNIFTY_1_log_sell_model.joblib',
    'nifty_5_buy_bilstm': './models/nifty_models_5min/NIFTY_5_bilstm_model.keras',
    'nifty_5_buy_lstm': './models/nifty_models_5min/NIFTY_5_lstm_model.keras',
    'nifty_5_buy_log': './models/nifty_models_5min/NIFTY_5_log_model.joblib',
    'nifty_5_sell_bilstm': './models/nifty_models_5min_sell/NIFTY_5_bilstm_sell_model.keras',
    'nifty_5_sell_lstm': './models/nifty_models_5min_sell/NIFTY_5_lstm_sell_model.keras',
    'nifty_5_sell_log': './models/nifty_models_5min_sell/NIFTY_5_log_sell_model.joblib',
}

logger.info("Model paths configured.")

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
    df_diff = data[::-1].diff().fillna(0)[::-1]
    logger.debug(f"Difference columns calculated. Data shape: {df_diff.shape}")
    return df_diff

def scaling(df):
    logger.info("Scaling data...")
    features = df.columns.tolist()
    scaler = MinMaxScaler(feature_range=(-100, 100))
    scaled_data = df[features].apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)
    sequence_reversed = scaled_data.values.tolist()[::-1]
    logger.debug(f"Data scaling completed. Data shape: {len(sequence_reversed)} sequences")
    return sequence_reversed

def prepare_data(data):
    logger.info("Preparing data...")
    data_with_indicators = add_technical_indicators(data)
    data_processed = calculate_new_columns(data_with_indicators)
    data_diff = calculate_diff_columns(data_processed)
    data_diff = scaling(data_diff)
    data_processed = scaling(data_processed)
    X1 = [data_processed]
    X2 = [data_diff]
    logger.debug(f"Data prepared. Data shapes: {np.array(X1).shape}, {np.array(X2).shape}")
    return X1, X2

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/memory')
def memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return f"RSS: {mem_info.rss / (1024 ** 2):.2f} MB, VMS: {mem_info.vms / (1024 ** 2):.2f} MB"

def process_request(csv_data, direction, model_type):
    logger.info(f"Processing request for {model_type}...")
    
    # Read and prepare data
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)
    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    
    result = False
    
    if direction == 1:
        logger.info("Processing buy direction...")
        xxc = load_model_on_demand(model_paths[f'{model_type}_buy_bilstm']).predict(data_without_diff)
        xxl = load_model_on_demand(model_paths[f'{model_type}_buy_lstm']).predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = load_joblib_model_on_demand(model_paths[f'{model_type}_buy_log']).predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        logger.info("Processing sell direction...")
        xxc = load_model_on_demand(model_paths[f'{model_type}_sell_bilstm']).predict(data_without_diff)
        xxl = load_model_on_demand(model_paths[f'{model_type}_sell_lstm']).predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = load_joblib_model_on_demand(model_paths[f'{model_type}_sell_log']).predict_proba(xx)[:, 1]
        result = pred > 0.5
    
    logger.debug(f"Prediction result: {result}")
    
    # Free up memory
    del df, data_without_diff, data_with_diff, xxc, xxl, xx, pred
    gc.collect()
    
    return jsonify(result.tolist())


@app.route('/nifty_1', methods=['POST'])
def nifty_1():
    data = request.get_json()
    return process_request(data.get('csvData'), data.get('dir'), 'nifty_1')

@app.route('/nifty_5', methods=['POST'])
def nifty_5():
    data = request.get_json()
    return process_request(data.get('csvData'), data.get('dir'), 'nifty_5')

@app.route('/nifty1_1', methods=['POST'])
def nifty1_1():
    data = request.get_json()
    return process_request(data.get('csvData'), data.get('dir'), 'nifty1_1')

@app.route('/nifty1_5', methods=['POST'])
def nifty1_5():
    data = request.get_json()
    return process_request(data.get('csvData'), data.get('dir'), 'nifty1_5')

@app.route('/banknifty_1', methods=['POST'])
def banknifty_1():
    data = request.get_json()
    return process_request(data.get('csvData'), data.get('dir'), 'banknifty_1')


@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"An error occurred: {str(e)}")
    return jsonify({"error": "An error occurred. Please try again later."}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8080))
#     app.run(host='0.0.0.0', port=port)
