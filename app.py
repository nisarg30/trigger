from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import talib
import joblib
import io
from keras.models import load_model

# nifty 1
nifty_1_buy_bilstm = load_model('./models/nifty_models_1min/NIFTY_1_bilstm_model.keras')
nifty_1_buy_lstm  = load_model('./models/nifty_models_1min/NIFTY_1_lstm_model.keras')
nifty_1_buy_log  = joblib.load('./models/nifty_models_1min/NIFTY_1_log_model.joblib')

nifty_1_sell_bilstm  = load_model('./models/nifty_models_1min_sell/NIFTY_1_bilstm_sell_model.keras')
nifty_1_sell_lstm  = load_model('./models/nifty_models_1min_sell/NIFTY_1_lstm_sell_model.keras')
nifty_1_sell_log  = joblib.load('./models/nifty_models_1min_sell/NIFTY_1_log_sell_model.joblib')

#nifty 5
nifty_5_buy_bilstm = load_model('./models/nifty_models_5min/NIFTY_5_bilstm_model.keras')
nifty_5_buy_lstm  = load_model('./models/nifty_models_5min/NIFTY_5_lstm_model.keras')
nifty_5_buy_log  = joblib.load('./models/nifty_models_5min/NIFTY_5_log_model.joblib')

nifty_5_sell_bilstm  = load_model('./models/nifty_models_5min_sell/NIFTY_5_bilstm_sell_model.keras')
nifty_5_sell_lstm  = load_model('./models/nifty_models_5min_sell/NIFTY_5_lstm_sell_model.keras')
nifty_5_sell_log  = joblib.load('./models/nifty_models_5min_sell/NIFTY_5_log_sell_model.joblib')

#nifty1 1
nifty1_1_buy_bilstm = load_model('./models/nifty1_models_1min/NIFTY1!_1_bilstm_model.keras')
nifty1_1_buy_lstm  = load_model('./models/nifty1_models_1min/NIFTY1!_1_lstm_model.keras')
nifty1_1_buy_log  = joblib.load('./models/nifty1_models_1min/NIFTY1!_1_log_model.joblib')

nifty1_1_sell_bilstm  = load_model('./models/nifty1_models_1min_sell/NIFTY1!_1_bilstm_sell_model.keras')
nifty1_1_sell_lstm  = load_model('./models/nifty1_models_1min_sell/NIFTY1!_1_lstm_sell_model.keras')
nifty1_1_sell_log  = joblib.load('./models/nifty1_models_1min_sell/NIFTY1!_1_log_sell_model.joblib')

#nifty1 5
nifty1_5_buy_bilstm = load_model('./models/nifty1_models_5min/NIFTY1!_5_bilstm_model.keras')
nifty1_5_buy_lstm  = load_model('./models/nifty1_models_5min/NIFTY1!_5_lstm_model.keras')
nifty1_5_buy_log  = joblib.load('./models/nifty1_models_5min/NIFTY1!_5_log_model.joblib')

nifty1_5_sell_bilstm  = load_model('./models/nifty1_models_5min_sell/NIFTY1!_5_bilstm_sell_model.keras')
nifty1_5_sell_lstm  = load_model('./models/nifty1_models_5min_sell/NIFTY1!_5_lstm_sell_model.keras')
nifty1_5_sell_log  = joblib.load('./models/nifty1_models_5min_sell/NIFTY1!_5_log_sell_model.joblib')

app = Flask(__name__)

def add_technical_indicators(df):
    print(df.shape)
    close_pricesx = df['Close'].values
    close_prices = close_pricesx[::-1]
    
    # EMA
    ema_20 = talib.EMA(close_prices, timeperiod=20)
    ema_50 = talib.EMA(close_prices, timeperiod=50)
    # RSI, custom_objects=None, compile=True, safe_mode=True
    rsi = talib.RSI(close_prices, timeperiod=14)
    # MACD
    macd, signal_line, _ = talib.MACD(close_prices)
    # BBands
    upper_band, middle_band, lower_band = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
    
    df['ema_20'] = ema_20[::-1]
    df['ema_50'] = ema_50[::-1]
    df['rsi'] = rsi[::-1]
    df['macd'] = macd[::-1]
    df['signal_line'] = signal_line[::-1]
    df['upper_band'] = upper_band[::-1]
    df['middle_band'] = middle_band[::-1]
    df['lower_band'] = lower_band[::-1]

    df = df.dropna() 
    return df

def calculate_new_columns(df):
    df = df.copy()  
    df['wvff'] = df['wvf'] - df['rangeHigh']
    df['ema'] = df['ema_20'] - df['ema_50']
    df['cema'] = df['Close'] - df['ema_20']
    df['lowc'] = df['Close'] - df['lower_band']
    df['macdf'] = df['macd'] - df['signal_line']
    df['midlc'] = df['Close'] - df['middle_band']
    df['wvflf'] = df['wvfl'] - df['rangeHighl']
    df = df.drop(columns=['Timestamp','Volume','Open','Max','Min','wvf','rangeHigh','wvfl','rangeHighl','ema_20', 'ema_50','macd','signal_line','upper_band','lower_band','middle_band'])
    return df

def calculate_diff_columns(data):
    xx = data.copy()
    xx = xx[::-1]
    df_diff = xx.diff().fillna(0)
    df_diff = df_diff[::-1]

    return df_diff

def scaling(df):
    features = df.columns.tolist()
    scaler = MinMaxScaler(feature_range=(-100, 100))
    first_61_rows = df.iloc[:61]
    sequence_scaled = first_61_rows[features].apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0).values.tolist()
    sequence_reversed = sequence_scaled[::-1]
    return sequence_reversed

def prepare_data(data):
    data_with_indicators = add_technical_indicators(data)
    data_processed = calculate_new_columns(data_with_indicators)
    data_diff = calculate_diff_columns(data_processed.copy())
    data_diff = scaling(data_diff.copy())
    data_processed = scaling(data_processed.copy())

    X1 = []
    X1.append(data_processed)
    X2 = []
    X2.append(data_diff)
    return X1, X2

@app.route('/nifty_1', methods=['POST'])
def nifty_1():
    csv_data = request.get_json().get('csvData')
    direction = request.get_json().get('dir')
    df = pd.read_csv(io.StringIO(csv_data))
    
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)

    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    result = False

    if direction == 1:
        xxc = nifty_1_buy_bilstm.predict(data_without_diff)
        xxl = nifty_1_buy_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty_1_buy_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        xxc = nifty_1_sell_bilstm.predict(data_without_diff)
        xxl = nifty_1_sell_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty_1_sell_log.predict_proba(xx)[:, 1]
        result = pred > 0.5

    return jsonify(result.tolist())

@app.route('/nifty_5', methods=['POST'])
def nifty_5():
    csv_data = request.get_json().get('csvData')
    direction = request.get_json().get('dir')
    df = pd.read_csv(io.StringIO(csv_data))
    
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)

    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    result = False

    if direction == 1:
        xxc = nifty_5_buy_bilstm.predict(data_without_diff)
        xxl = nifty_5_buy_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty_5_buy_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        xxc = nifty_5_sell_bilstm.predict(data_without_diff)
        xxl = nifty_5_sell_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty_5_sell_log.predict_proba(xx)[:, 1]
        result = pred > 0.5

    return jsonify(result.tolist())

@app.route('/nifty1_1', methods=['POST'])
def nifty1_1():
    csv_data = request.get_json().get('csvData')
    direction = request.get_json().get('dir')
    df = pd.read_csv(io.StringIO(csv_data))
    
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)

    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    result = False

    if direction == 1:
        xxc = nifty1_1_buy_bilstm.predict(data_without_diff)
        xxl = nifty1_1_buy_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty1_1_buy_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        xxc = nifty1_1_sell_bilstm.predict(data_without_diff)
        xxl = nifty1_1_sell_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty1_1_sell_log.predict_proba(xx)[:, 1]
        result = pred > 0.5

    return jsonify(result.tolist())

@app.route('/nifty1_5', methods=['POST'])
def nifty1_5():
    csv_data = request.get_json().get('csvData')
    direction = request.get_json().get('dir')
    df = pd.read_csv(io.StringIO(csv_data))
    
    df = df.astype(float, errors='ignore')
    data_without_diff, data_with_diff = prepare_data(df)

    data_without_diff = np.array(data_without_diff)
    data_with_diff = np.array(data_with_diff)
    result = False

    if direction == 1:
        xxc = nifty1_5_buy_bilstm.predict(data_without_diff)
        xxl = nifty1_5_buy_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty1_5_buy_log.predict_proba(xx)[:, 1]
        result = pred > 0.5
    else:
        xxc = nifty1_5_sell_bilstm.predict(data_without_diff)
        xxl = nifty1_5_sell_lstm.predict(data_with_diff)
        xx = pd.DataFrame({'s': xxc.flatten(), 'l': xxl.flatten()})
        pred = nifty1_5_sell_log.predict_proba(xx)[:, 1]
        result = pred > 0.5

    return jsonify(result.tolist())

if __name__ == '__main__':
    app.run(debug=True)