# Retrieve 5m OHLC data on BTC/USD.
# Endpoint does not require authentication,
# but has utility functions for authentication.

import http.client
import urllib.request
import urllib.parse
import hashlib
import hmac
import base64
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from fastapi import FastAPI, Query
import os
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
import lightgbm
import torch
import torch.nn as nn

#---------------------------------------
# Functions
#---------------------------------------

class GRUModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

#---------------------------------------
# Data Extract
#---------------------------------------
def main():
    response = request(
        method="GET",
        path="/0/public/OHLC",
        query={
            "pair": "ETH/USD",
            "interval": 1440,
        },
        environment="https://api.kraken.com",
    )
    data = response.read().decode()
    json_data = json.loads(data)
    eth_aud = json_data["result"]["ETH/USD"]

    cols = ["time", "open", "high", "low", "close", "vwap", "volumn_f", "count"]
    df = pd.DataFrame(eth_aud, columns=cols)

    numeric_cols = ["open", "high", "low", "close", "vwap", "volumn_f"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    df["volume"] = df["vwap"] * df["volumn_f"]
    df = df.iloc[::-1].reset_index(drop=True)
    
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.floor('h')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df = df.groupby(["timestamp"]).agg(
        open=("open", "last"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "first"),
        volume=("volume", "sum")
    ).reset_index()
    
    df["marketCap"] = df["close"] * df["volume"]
    
    df = df.sort_index(ascending=True)

    return df

def request(method: str = "GET", path: str = "", query: dict | None = None, body: dict | None = None, public_key: str = "", private_key: str = "", environment: str = "") -> http.client.HTTPResponse:
    url = environment + path
    query_str = ""
    if query is not None and len(query) > 0:
        query_str = urllib.parse.urlencode(query)
        url += "?" + query_str
        nonce = ""
    if len(public_key) > 0:
        if body is None:
            body = {}
        nonce = body.get("nonce")
        if nonce is None:
            nonce = get_nonce()
            body["nonce"] = nonce
    headers = {}
    body_str = ""
    if body is not None and len(body) > 0:
        body_str = json.dumps(body)
        headers["Content-Type"] = "application/json"
    if len(public_key) > 0:
        headers["API-Key"] = public_key
        headers["API-Sign"] = get_signature(private_key, query_str+body_str, nonce, path)
    req = urllib.request.Request(
        method=method,
        url=url,
        data=body_str.encode(),
        headers=headers,
    )
    return urllib.request.urlopen(req)

def get_nonce() -> str:
    return str(int(time.time() * 1000))

def get_signature(private_key: str, data: str, nonce: str, path: str) -> str:
    return sign(
        private_key=private_key,
        message=path.encode() + hashlib.sha256(
            (nonce + data)
            .encode()
        ).digest()
    )

def sign(private_key: str, message: bytes) -> str:
    return base64.b64encode(
        hmac.new(
            key=base64.b64decode(private_key),
            msg=message,
            digestmod=hashlib.sha512,
        ).digest()
    ).decode()
    
#---------------------------------------
# Load Artifact
#---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.getcwd()  # current directory
joblib_folder = os.path.join(current_dir, "api", "models", "13475823_Siheng")
model_path = os.path.join(joblib_folder, "gru_checkpoint.pth")
scaler_path = os.path.join(joblib_folder, "scaler.pkl")

scaler_X = joblib.load(scaler_path)
checkpoint = torch.load(model_path, map_location=device)

#---------------------------------------
# Build API
#---------------------------------------

app = FastAPI(title="Crypto Currency Next Day Prediction")

@app.get("/")
def root():
    df = main()
    df = df.reset_index(drop=True)
    return json.loads(df.to_json(orient="records"))

@app.post("/predict")
def predict(
    start_date: str | None = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str | None = Query(None, description="End date in YYYY-MM-DD format"),
    period: str | None = Query(None, description="Optional quick period filter: 1w, 1m, 1y, 3y")
):
    df = main()
    df = df.reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Time filtering logic ---
    latest_date = df["timestamp"].max()

    if period:
        period = period.lower().strip()
        if period == "1w":
            start = latest_date - timedelta(weeks=1)
        elif period == "1m":
            start = latest_date - timedelta(days=30)
        elif period == "1y":
            start = latest_date - timedelta(days=365)
        elif period == "3y":
            start = latest_date - timedelta(days=3 * 365)
        else:
            start = df["timestamp"].min()
        df = df[df["timestamp"] >= start]
    elif start_date and end_date:
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        except Exception:
            return {"error": "Invalid date format. Please use YYYY-MM-DD."}
    else:
        df = main()[-7:]

    # --- Preprocessing ---
    source_df = df.copy()
    source_df['open_day'] = source_df['timestamp'].dt.day
    source_df['open_year'] = source_df['timestamp'].dt.year
    source_df['open_month'] = source_df['timestamp'].dt.month
    source_df = source_df.drop(columns = ["timestamp"])
    for col in ['open', 'close', 'high', 'low']:
        source_df[f'log_{col}'] = np.log(source_df[col])
    source_df['log_return'] = source_df['log_close'].diff()
    source_df['lag_return_1'] = source_df['log_close'].diff(1)
    source_df['lag_return_3'] = source_df['log_close'].diff(3)
    source_df['rolling_vol_3'] = source_df['close'].rolling(window=3).std() / source_df['close']
    ema12 = source_df['close'].ewm(span=12, adjust=False).mean()
    ema26 = source_df['close'].ewm(span=26, adjust=False).mean()
    source_df['MACD'] = ema12 - ema26
    source_df['MACD_signal'] = source_df['MACD'].ewm(span=9, adjust=False).mean()
    window = 14
    delta = source_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    source_df['volume_change'] = source_df['volume'].pct_change()
    # source_df['y_target'] = source_df['log_high'].shift(-1).rolling(3).mean()
    source_df = source_df.dropna()
    source_df = source_df.drop('high', axis=1)

    scaled_values = scaler_X.transform(source_df.values)

    sequence_length = 60  # same as training

    n_features = 19  # update this to match your feature count
    model = GRUModel(n_features=n_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    # Make prediction
    future_predictions = []
    current_seq = scaled_values[-sequence_length:]

    last_log_high = source_df['log_high'].iloc[-1]

    for _ in range(7):  # predict next 7 steps
        seq_tensor = torch.tensor(current_seq[np.newaxis, :, :], dtype=torch.float32)
        
        with torch.no_grad():
            pred = model(seq_tensor).cpu().numpy().flatten()[0]
        
        last_log_high = source_df['log_high'].iloc[-1] if len(future_predictions) == 0 else next_log_high
        next_log_high = last_log_high + pred
        next_high = np.expm1(next_log_high)
        
        future_predictions.append(next_high)
        
        # Append predicted features back into sequence (optional approximation)
        new_row = current_seq[-1].copy()
        new_row[0] = next_log_high  # update log_close-like feature
        current_seq = np.vstack([current_seq[1:], new_row])

        last_log_high = next_log_high


    # --- Output ---
    return {
        "start_date": str(df["timestamp"].min().date()),
        "end_date": str(df["timestamp"].max().date()),
        "7d_prediction": future_predictions,
        "next_day_prediciton": float(np.expm1(last_log_high))
    }