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
import pandas as pd
from fastapi import FastAPI
import os
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib

#---------------------------------------
# Functions
#---------------------------------------

def log_tran(df):
    for col in ['volume', 'marketCap', 'volume_to_mcap']:
        if col in df.columns:
            df[col] = np.log1p(df[col].fillna(0))
    return df

def price_change(df):
    df['price_change'] = df['close'] - df['open']
    return df

def price_change_pct(df):
    df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
    return df

def volume_to_mcap(df):
    df['volume_to_mcap'] = df['volume'] / df['marketCap']
    return df



#---------------------------------------
# Data Extract
#---------------------------------------
def main():
    response = request(
        method="GET",
        path="/0/public/OHLC",
        query={
            "pair": "ETH/AUD",
            "interval": 5,
        },
        environment="https://api.kraken.com",
    )
    data = response.read().decode()
    json_data = json.loads(data)
    eth_aud = json_data["result"]["ETH/AUD"]

    cols = ["time", "open", "high", "low", "close", "vwap", "volumn_f", "count"]
    df = pd.DataFrame(eth_aud, columns=cols)

    numeric_cols = ["open", "high", "low", "close", "vwap", "volumn_f"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df["volume"] = df["vwap"] * df["volumn_f"]
    
    # Reverse for proper aggregation: latest → first
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Label each 24-row block as 1 day backward from most recent
    # df["day_index"] = df.index // 24
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['open_year'] = df['time'].dt.year
    df['open_month'] = df['time'].dt.month
    df['open_day'] = df['time'].dt.day
    df['open_hour'] = df['time'].dt.hour
    
    # Aggregate daily OHLCV
    daily_df = df.groupby(["open_year", "open_month", "open_day", "open_hour"]).agg({
        "open": "last",    # since reversed
        "high": "max",
        "low": "min",
        "close": "first",  # since reversed
        "volume": "sum"
    })
    
    # Compute daily Market Cap (using closing price × total volume)
    daily_df["marketCap"] = daily_df["close"] * daily_df["volume"]
    
    # Optional: rename so day_index 0 = most recent day
    daily_df = daily_df.sort_index(ascending=True)

    return daily_df

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
# Build API
#---------------------------------------
current_dir = os.getcwd()  # current directory
joblib_folder = os.path.join(current_dir, "api", "models", "13475823_Siheng")
model_path = os.path.join(joblib_folder, "catboost_model.joblib")
tf_path = os.path.join(joblib_folder, "scaler.joblib")

scaler = joblib.load(tf_path)
model = joblib.load(model_path)

app = FastAPI(title="CatBoost Regression API")

@app.post("/predict")
def predict():
    td_df = main()
    df = df.reset_index(drop=True)

    for col in ['volume', 'marketCap', 'volume_to_mcap']:
        if col in df.columns:
            df[col] = np.log1p(df[col].fillna(0))
    
    for col in ['open_month', 'open_year', 'open_day', 'open_hour']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    df['price_change'] = df['close'] - df['open']
    df['volume_to_mcap'] =df['volume'] / df['marketCap']
    df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100


    scaled_df = scaler.transform(df)
    pred = model.predict(scaled_df)

    return {"prediction": float(pred[0])}
