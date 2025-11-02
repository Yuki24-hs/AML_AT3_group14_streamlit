import json
import time
import urllib
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import requests
import streamlit as st
from requests.exceptions import RequestException

# Define variables for API endpoints
API_BASE = "https://at3-api-25229384-latest.onrender.com/"
API_HEALTH_POINT = "health/"
API_PREDICT = "predict/bitcoin/"
API_PREDICT7D = "predict/bitcoin/7d"

csv_path = "data/25229384-AT3-Bitcoin.csv"


def get_kraken_api_data(pair="BTC/USD", interval=1440):
    # Build request
    base = "https://api.kraken.com"
    path = "/0/public/OHLC"
    url = f"{base}{path}?{urllib.parse.urlencode({'pair': pair, 'interval': interval})}"

    # Call Kraken API
    req = urllib.request.Request(method="GET", url=url)
    with urllib.request.urlopen(req) as resp:
        payload = resp.read().decode()

    data = json.loads(payload)
    if data.get("error"):
        raise RuntimeError(f"Kraken API Error: {data['error']}")

    # Get dynamic result key (e.g. 'BTCUSD', 'XXBTZUSD')
    result = data.get("result", {})
    keys = [k for k in result.keys() if k != "last"]
    if not keys:
        raise RuntimeError("No OHLC data found in Kraken response.")
    pair_key = keys[0]

    # Build DataFrame
    df = pd.DataFrame(
        result[pair_key],
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "count",
        ],
    )

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    for col in ["open", "high", "low", "close", "vwap", "volume", "count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add marketCap = vwap * volume
    df["marketCap"] = df["vwap"] * df["volume"]

    # Rename and return consistent format
    df = df.rename(columns={"timestamp": "date"})
    return df[["date", "open", "high", "low", "close", "volume", "marketCap"]]


def get_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load historical OHLC crypto data from a CSV.
    Expects at least: 'timeOpen', 'open', 'high', 'low', 'close', 'volume', 'marketCap'
    Returns: ['date','open','high','low','close','volume','marketCap']
    """
    df = pd.read_csv(file_path)

    # Convert timeOpen to datetime
    df["date"] = pd.to_datetime(df["timeOpen"], utc=True, errors="coerce")

    # Keep only required columns
    df = df[["date", "open", "high", "low", "close", "volume", "marketCap"]]

    # Clean and sort
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=3600)  # cache for 1 hour
def combine_data(
    file_path: str, pair: str = "BTC/USD", interval: int = 1440
) -> pd.DataFrame:
    # API data
    api_data = get_kraken_api_data(pair=pair, interval=interval)

    # CSV data
    csv_data = get_csv_data(file_path)

    # Keep CSV rows strictly before earliest API date
    min_api_date = api_data["date"].min()
    csv_prior = csv_data[csv_data["date"] < min_api_date].copy()

    # Concat, sort, and de-dup
    final = pd.concat([csv_prior, api_data], ignore_index=True)
    final = (
        final.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    # Ensure expected column order
    final = final[["date", "open", "high", "low", "close", "volume", "marketCap"]]

    return final


@st.cache_data(ttl=3600, show_spinner=False)  # cache for 1 hour
def get_7d_data(api_base):

    try:
        response = requests.get(api_base, timeout=800)
        response.raise_for_status()
        data = response.json()

        # Expected format: list of dicts
        if not isinstance(data, list):
            raise ValueError("Unexpected response format: expected a list of items")

        df = pd.DataFrame(data)
        df["predicted_date"] = pd.to_datetime(df["predicted_date"], errors="coerce")
        df["predicted_high"] = pd.to_numeric(df["predicted_high"], errors="coerce")
        return df

    except Exception as e:
        st.error(f"⚠️ Failed to fetch 7-day data: {e}")
        return pd.DataFrame(columns=["predicted_date", "predicted_high"])


@st.cache_data(ttl=3600, show_spinner=False)  # cache for 1 hour
def get_next_day_prediction(api_base, date=None):
    """
    Call API return a one-row DataFrame:['predicted_date', 'predicted_high'].
    """
    try:
        resp = requests.get(api_base, timeout=800)
        resp.raise_for_status()
        data = resp.json()

        # predicted_date (str YYYY-MM-DD), predicted_high (float)
        predicted_date = data.get("predicted_date")
        predicted_high = data.get("predicted_high", None)

        df = pd.DataFrame(
            [
                {
                    "date": pd.to_datetime(predicted_date, errors="coerce"),
                    "high": pd.to_numeric(predicted_high, errors="coerce"),
                }
            ]
        )
        return df

    except Exception as e:
        st.error(f"⚠️ Failed to fetch next-day prediction: {e}")
        return pd.DataFrame(columns=["date", "high"])


# Fetch prediction (cached for 5 minutes)
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prediction(api_base):
    """
    Fetch Bitcoin next-day prediction date from FastAPI (Render-hosted).
    Cached for 5 minutes.
    """
    try:
        resp = requests.get(api_base, timeout=800)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"⚠️ API request failed: {e}")
        return None


def format_millions(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return str(value)


# get the cached combined data
combined_data = combine_data(csv_path)

# Get the last two days
current_high = combined_data.iloc[-1]["high"]
previous_high = combined_data.iloc[-2]["high"]

current_low = combined_data.iloc[-1]["low"]
previous_low = combined_data.iloc[-2]["low"]

current_open = combined_data.iloc[-1]["open"]
previous_open = combined_data.iloc[-2]["open"]

current_close = combined_data.iloc[-1]["close"]
previous_close = combined_data.iloc[-2]["close"]

current_volume = combined_data.iloc[-1]["volume"]
previous_volume = combined_data.iloc[-2]["volume"]

current_marketCap = combined_data.iloc[-1]["marketCap"]
previous_marketCap = combined_data.iloc[-2]["marketCap"]


def run():
    st.markdown(
        """
        <style>
        h1 {
            font-size: 2.2rem !important;   /* Title font */
            font-weight: 700 !important;
            margin-bottom: 0.5rem;
        }
        .stMarkdown p {
            font-size: 1.3rem !important;   /* Description text */
            line-height: 1.6;
        }
        .predicted-date {
            font-size: 1.3rem !important;   /* Predicted date text */
            color: #0f172a;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Page Title
    st.title("Bitcoin BTC Dashboard")
    st.markdown(
        """Bitcoin was launched in 2009 by Satoshi Nakamoto, marking the beginning of decentralized digital currency. It is powered by blockchain technology, a secure and transparent distributed ledger that records transactions across a network of computers, ensuring trust without intermediaries. Users can buy Bitcoin through cryptocurrency exchanges, store it in digital wallets, and use it for online transactions, long-term investments, or as a hedge against inflation. Over the years, Bitcoin has experienced massive growth — evolving from an experimental concept to a trillion-dollar asset class, often referred to as “digital gold.” Today, it stands as a global financial phenomenon, transforming the way people perceive money, value, and financial independence."""
    )

    # Fetch Automatically on Page Load
    with st.spinner("Fetching predicted date from FastAPI..."):
        data = fetch_prediction(API_BASE + API_PREDICT)
        if data:
            predicted_date = data.get("predicted_date", "N/A")
            st.markdown(f"**🕒 Current Time Zone for Data:** {predicted_date}")
        else:
            st.warning("No data received from the API.")

    st.markdown("""## Financial Predictors""")

    with st.container():
        cols = st.columns(6, gap="medium")

        with cols[0]:
            st.metric(
                "High",
                f"{current_high:0.1f}",
                delta=f"{(((current_high - previous_high)/previous_high)*100):0.2f} %",
            )

        with cols[1]:
            st.metric(
                "Low",
                f"{current_low:0.1f}",
                delta=f"{(((current_low - previous_low)/previous_low)*100):0.2f} %",
            )

        with cols[2]:
            st.metric(
                "Open",
                f"{current_open:0.1f}",
                delta=f"{(((current_open - previous_open)/previous_open)*100):0.2f} %",
            )

        with cols[3]:
            st.metric(
                "Close",
                f"{current_close:0.1f}",
                delta=f"{(((current_close - previous_close)/previous_close)*100):0.2f} %",
            )

        with cols[4]:
            st.metric(
                "Volume",
                f"{current_volume:0.1f}",
                delta=f"{(((current_volume - previous_volume)/previous_volume)*100):0.2f} %",
            )

        with cols[5]:
            st.metric(
                "Market Capital",
                f"{format_millions(current_marketCap)}",
                delta=f"{(((current_marketCap - previous_marketCap)/previous_marketCap)*100):0.2f} %",
            )

        cols2 = st.columns([3, 1], gap="medium")

        with cols2[0]:
            TIMEFRAME = ["1W", "1M", "6M", "1Y", "3Y", "5Y", "Max"]
            tabs = st.tabs(TIMEFRAME)

            today = combined_data["date"].max()

            tab_ranges = {
                "1W": today - timedelta(days=7),
                "1M": today - timedelta(days=30),
                "6M": today - timedelta(days=182),
                "1Y": today - timedelta(days=365),
                "3Y": today - timedelta(days=1095),
                "5Y": today - timedelta(days=1825),
                "Max": combined_data["date"].min(),
            }

            for tab_name, tab in zip(tab_ranges.keys(), tabs):
                with tab:
                    start_date = tab_ranges[tab_name]
                    filtered_df = combined_data[combined_data["date"] >= start_date]

                    if filtered_df.empty:
                        st.warning("No data available for this period.")
                    else:
                        chart = (
                            alt.Chart(filtered_df)
                            .mark_line(point=False)
                            .encode(
                                x="date:T",
                                y="close:Q",
                                tooltip=[
                                    alt.Tooltip("date:T", title="Date"),
                                    alt.Tooltip(
                                        "open:Q",
                                        title="Open",
                                        format=".2f",
                                        formatType="number",
                                    ),
                                    alt.Tooltip(
                                        "high:Q",
                                        title="High",
                                        format=".2f",
                                        formatType="number",
                                    ),
                                    alt.Tooltip(
                                        "low:Q",
                                        title="Low",
                                        format=".2f",
                                        formatType="number",
                                    ),
                                    alt.Tooltip(
                                        "close:Q",
                                        title="Close",
                                        format=".2f",
                                        formatType="number",
                                    ),
                                ],
                            )
                            .interactive()
                        )  # zoom & pan

                        st.altair_chart(chart, use_container_width=True)
        with cols2[1]:

            price_change = current_high - previous_high
            delta_percent = (price_change / previous_high) * 100
            pc_color = "green" if price_change >= 0 else "red"
            arrow = "▲" if price_change >= 0 else "▼"

    st.divider()
    # section for forecasts
    st.markdown("""## Financial Forecasts""")

    disclaimer = "This project is developed for academic and research purposes only. Cryptocurrency markets are inherently volatile and speculative; therefore, model forecasts should not be used as the sole basis for any trading or investment decision. The author and affiliated institution accept no liability for financial losses or actions taken based on the results of this study."
    st.warning(disclaimer, icon="⚠️")

    with st.spinner(" Loading predictions..."):
        # get data and then display the charts
        predicted_data = get_7d_data(API_BASE + API_PREDICT7D)
        predicted_data["predicted_date"] = pd.to_datetime(
            predicted_data["predicted_date"]
        ).dt.date
        prediction_next_day = get_next_day_prediction(API_BASE + API_PREDICT, date=None)
        prediction_next_day["date"] = prediction_next_day["date"].dt.date

        # get last 7 rows of data from combined data
        real_data = combined_data.tail(7).copy()
        real_data = real_data[["date", "high"]]
        real_data["date"] = real_data["date"].dt.date

        # create a df for 2nd chart
        next_df = pd.concat([real_data, prediction_next_day], ignore_index=True)
        next_df = next_df.sort_values("date").reset_index(drop=True)
        # only keep last two of next_df
        next_df = next_df.tail(2).copy()

        # create two columns
        cols_pred = st.columns(2, gap="medium")

        # for 7 days pred
        y_min = min(real_data["high"].min(), predicted_data["predicted_high"].min())
        y_max = max(real_data["high"].max(), predicted_data["predicted_high"].max())

        # for next day pred
        real_data_nd = real_data.tail(6).copy()
        y_min_nd = min(real_data_nd["high"].min(), next_df["high"].min())
        y_max_nd = max(real_data_nd["high"].max(), next_df["high"].max())

        with cols_pred[0]:
            # plot the actual vs predicted for 7 days
            real_line = (
                alt.Chart(real_data)
                .mark_line(color="#1f77b4")
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y(
                        "high:Q",
                        title="High Price",
                        scale=alt.Scale(domain=[y_min, y_max]),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("high:Q", title="Actual High", format=".2f"),
                    ],
                )
            )
            pred_line = (
                alt.Chart(predicted_data)
                .mark_line(strokeDash=[5, 5], color="#ff7f0e")
                .encode(
                    x=alt.X("predicted_date:T", title="predicted_date"),
                    y=alt.Y(
                        "predicted_high:Q",
                        title="High Price",
                        scale=alt.Scale(domain=[y_min, y_max]),
                    ),
                    tooltip=[
                        alt.Tooltip("predicted_date:T", title="Timestamp"),
                        alt.Tooltip(
                            "predicted_high:Q", title="Predicted High", format=".2f"
                        ),
                    ],
                )
            )
            combined = (
                alt.layer(real_line, pred_line)
                .properties(width="container", height=400)
                .configure_legend(orient="bottom")
            )
            combined = (
                (real_line + pred_line)
                .properties(width="container", height=400)
                .configure_legend(orient="bottom")
            )
            with st.container(border=False):
                st.altair_chart(combined, use_container_width=True)

        with cols_pred[1]:
            real_line = (
                alt.Chart(real_data_nd)
                .mark_line(color="#1f77b4")
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y(
                        "high:Q",
                        title="High Price",
                        scale=alt.Scale(domain=[y_min_nd, y_max_nd]),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("high:Q", title="Actual High", format=".2f"),
                    ],
                )
            )
            pred_line = (
                alt.Chart(next_df)
                .mark_line(color="#ff7f0e", strokeDash=[5, 5])
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y(
                        "high:Q",
                        title="High Price",
                        scale=alt.Scale(domain=[y_min_nd, y_max_nd]),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("high:Q", title="Predicted High", format=".2f"),
                    ],
                )
            )
            chart = (
                alt.layer(real_line, pred_line)
                .properties(width="container", height=400)
                .configure_legend(orient="bottom")
            )
            with st.container(border=False):
                st.altair_chart(chart, use_container_width=True)

        # create two columns to access predictions and to display dataframe

    st.dataframe(combined_data[:10], use_container_width=True)


# if __name__ == "__main__":
#     run()
