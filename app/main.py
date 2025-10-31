# app/main.py
from __future__ import annotations

import importlib
import sys
import os
import optparse
from pathlib import Path
import pandas as pd
import requests
import altair as alt
from datetime import datetime, timedelta
import http.client
import urllib.request
import urllib.parse
import hashlib
import hmac
import base64
import json
import time

import streamlit as st

# page config
st.set_page_config(
    page_title="AT3 Group 14", layout="wide", initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent  # AT3_GROUP14/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from students import s13475823_Siheng

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Main", "Student 1", "ETH", "Student 3", "Student 4"],
    label_visibility="collapsed",
)

MODULE_MAP = {
    "Bitcoin": "students.25229384_Yukthi",
    "Ethereum": "students.13475823_Siheng",
    "XRP": "students.25658663_Queenie",
    "Solana": "students.25428006_Mitali",
}

# Functions
def aggregate_crypto_data(df):
    """
    Aggregate crypto data by 'crypto' with total volume, total market cap,
    average high price, and price change over 1 hour and 24 hours.
    """
    if not {"timestamp", "crypto", "high", "volume", "marketCap"}.issubset(df.columns):
        raise KeyError("DataFrame must include ['timestamp', 'crypto', 'high', 'volume', 'marketCap'] columns.")

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort by crypto and timestamp
    df = df.sort_values(["crypto", "timestamp"])

    # --- Calculate percentage changes ---
    df["high_change_1h"] = df.groupby("crypto")["high"].pct_change(periods=1) * 100
    df["high_change_24h"] = df.groupby("crypto")["high"].pct_change(periods=24) * 100

    # --- Aggregate metrics ---
    agg_df = (
        df.groupby("crypto")
        .agg(
            change_1h=("high_change_1h", "mean"),
            change_24h=("high_change_24h", "mean"),
            high=("high", "mean"),
            volume=("volume", "mean"),
            marketCap=("marketCap", "mean")
        )
        .reset_index()
    )

    return agg_df

def fetch_crypto_data(pair: str, environment="https://api.kraken.com") -> pd.DataFrame:
    """Fetch OHLC data for a given crypto pair and return a processed DataFrame."""
    response = request(
        method="GET",
        path="/0/public/OHLC",
        query={"pair": pair, "interval": 1440},  # 1-day interval
        environment=environment,
    )

    data = response.read().decode()
    json_data = json.loads(data)
    result = json_data["result"]

    # Some Kraken responses include "last" key, we filter it out
    for key in result:
        if key != "last":
            records = result[key]
            break

    cols = ["timestamp", "open", "high", "low", "close", "vwap", "volume_f", "count"]
    df = pd.DataFrame(records, columns=cols)

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "vwap", "volume_f"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Compute additional columns
    df["volume"] = df["vwap"] * df["volume_f"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.iloc[::-1].reset_index(drop=True)

    # Aggregate daily OHLC
    df = (
        df.groupby(["timestamp"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )

    # Compute market cap (close × volume)
    df["marketCap"] = df["close"] * df["volume"]

    # Add crypto name (e.g., "ETH/USD" → "ETH")
    df["crypto"] = pair.split("/")[0]

    return df


def request(method="GET", path="", query=None, body=None, public_key="", private_key="", environment=""):
    """HTTP request wrapper for Kraken public API."""
    url = environment + path
    if query:
        query_str = urllib.parse.urlencode(query)
        url += "?" + query_str
    headers = {}
    body_str = json.dumps(body) if body else None
    if body_str:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(method=method, url=url, data=body_str.encode() if body_str else None, headers=headers)
    return urllib.request.urlopen(req)


# ---- MAIN FUNCTION ----
def main():
    cryptos = ["ETH/USD", "BTC/USD", "SOL/USD", "XRP/USD"]
    dfs = []

    for pair in cryptos:
        try:
            df = fetch_crypto_data(pair)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to fetch {pair}: {e}")

    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(["crypto", "timestamp"]).reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()

def filter_by_range(df):
    """Allow user to filter dataframe by date range via dropdown."""

    option = st.selectbox("Select Time Range:", ("All", "1 Week", "1 Month", "1 Year"), index=0)

    latest_date = df["timestamp"].max()
    if option == "1 Week":
        df = df[df["timestamp"] >= latest_date - timedelta(weeks=1)]
    elif option == "1 Month":
        df = df[df["timestamp"] >= latest_date - timedelta(days=30)]
    elif option == "1 Year":
        df = df[df["timestamp"] >= latest_date - timedelta(days=365)]

    return df

def filter_by_crypto(df):
    """Filter dataframe by cryptocurrency type using checkboxes."""

    st.markdown("Select Cryptocurrency")
    if "crypto" not in df.columns:
        st.error("The dataframe is missing a 'crypto' column.")
        return df

    crypto_types = sorted(df["crypto"].unique())
    selected = []

    # Display a checkbox for each crypto
    cols = st.columns(len(crypto_types))
    for i, crypto in enumerate(crypto_types):
        with cols[i]:
            if st.checkbox(crypto, value=True):
                selected.append(crypto)

    # Filter dataframe based on selected cryptos
    if selected:
        df = df[df["crypto"].isin(selected)]
    else:
        st.warning("No cryptocurrency selected. Displaying empty data.")
        df = df.iloc[0:0]  # return empty DataFrame

    return df


def plot_multi_crypto_high(df):
    """Plot high price trendlines for multiple cryptocurrencies."""
    if df.empty:
        st.warning("No data available for plotting.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    y_min, y_max = df["high"].min(), df["high"].max()
    hover = alt.selection_point(fields=["timestamp"], nearest=True, on="mouseover", empty=False)

    base = alt.Chart(df).encode(
        x=alt.X("timestamp:T", title="Date"),
        y=alt.Y("high:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color("crypto:N", title="Cryptocurrency", legend=alt.Legend(orient="top")),
    )

    # Smooth line for each crypto
    line = base.mark_line(interpolate="monotone")

    # Hover points
    points = base.mark_circle(size=60).encode(
        tooltip=[
            alt.Tooltip("timestamp:T", title="Timestamp"),
            alt.Tooltip("crypto:N", title="Crypto"),
            alt.Tooltip("high:Q", title="High", format=".2f"),
            alt.Tooltip("close:Q", title="Close", format=".2f"),
            alt.Tooltip("low:Q", title="Low", format=".2f"),
            alt.Tooltip("volume:Q", title="Volume", format=".2f"),
            alt.Tooltip("marketCap:Q", title="Market Cap", format=".2f"),
        ],
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    ).add_params(hover)

    chart = (line + points).properties(
        width="container", height=400
    )

    st.altair_chart(chart, use_container_width=True)

def plot_market_cap(df):
    """Plot daily market cap bar chart."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("timestamp:T", title="Date"),
            y=alt.Y("marketCap:Q", title="Market Cap")
        )
        .properties(width=800, height=300)
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )
    st.subheader("Daily Market Cap")
    st.altair_chart(chart, use_container_width=True)


# Main page content
def show_main():
    st.markdown("# AT3 Group 14 Project Overview")
    st.write(
        """
        **Project:** 

        **Goal:** Business problem

        **Dataset(s):** Optional.

        **Methods & Stack:**
        - Modeling: (e.g., Random Forest, XGBoost, Logistic Regression)
        - App: Streamlit
        - Backend / APIs: Fast API
        - Infra: (Docker/Poetry)

        **Team:** 
        - Member 1: Mitali H Balki ( 25428006 ) 
        - Member 2: Siheng Li ( 13475823)
        - Member 3: Queenie Goh ( XXX )
        - Member 4: Yukthi Hosadurga Shivalingegowda ( 25229384 )

        **Next steps / risks:** Later.
        """
    )
    df = main()
    st.markdown("## Market Trend Lines")
    df = filter_by_range(df)
    df = filter_by_crypto(df)
    plot_multi_crypto_high(df)
    agg_df = aggregate_crypto_data(df)
    st.dataframe(agg_df, use_container_width=True)

if page == "Main":
    show_main()
elif page == "ETH":
    s13475823_Siheng.run()
else:
    module_path = MODULE_MAP[page]
    try:
        mod = importlib.import_module(module_path)
        if hasattr(mod, "run") and callable(mod.run):
            mod.run()
        else:
            st.error(
                f"`{module_path}.run()` not found. Add a `run()` function in that file."
            )
    except ModuleNotFoundError as e:
        st.error(f"Module not found: {module_path}\n\n" f"Details: {e}")
