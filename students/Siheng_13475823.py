import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import datetime, timedelta, timezone


API_URL = "https://aml-eth-api-13475823.onrender.com"
API_URL_PREDICT = f"{API_URL}/predict"

st.set_page_config(page_title="Crypto Price Prediction", page_icon="💰", layout="wide")

# ---------- DATA FETCHING ----------
@st.cache_data(ttl=60)
def fetch_data():
    """Fetch data from API and return as a pandas DataFrame."""
    try:
        response = requests.get(f"{API_URL}/")
        response.raise_for_status()
        return pd.DataFrame.from_records(response.json())
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None


# ---------- Function & Helper ----------
def preprocess_data(df):
    """Convert timestamp and sort values."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("timestamp")

def format_large_number(num):
    """Format large numbers with units K (thousand), M (million), or B (billion)."""
    if abs(num) >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"${num / 1_000:.2f}K"
    else:
        return f"${num:.2f}"

def fetch_prediction(period):
    """Fetch prediction data from API with a given period."""
    try:
        response = requests.post(API_URL_PREDICT, params={"period": period}, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling prediction API: {e}")
        return None

# ---------- Trend Line Chart ----------

def filter_by_range(df):
    """Interactive date range filter styled like the new Streamlit template, compatible with v1.36.0."""
    st.markdown("### ⏳ Date Range Filter")
    # Use radio buttons (pills are not available before v1.38)
    option = st.radio(
        "Select a time period to view corresponding cryptocurrency data.",
        options=["All", "1 Week", "1 Month", "1 Year"],
        index=0,
        horizontal=True
    )
    latest_date = df["timestamp"].max()
    if option == "1 Week":
        df = df[df["timestamp"] >= latest_date - timedelta(weeks=1)]
    elif option == "1 Month":
        df = df[df["timestamp"] >= latest_date - timedelta(days=30)]
    elif option == "1 Year":
        df = df[df["timestamp"] >= latest_date - timedelta(days=365)]
    return df

def plot_high_price(df):
    """Display a high price line chart with hover tooltip inside a bordered section."""
    st.markdown("### 📈 High Price Overview")
    y_min, y_max = df["high"].min(), df["high"].max()
    hover = alt.selection_point(fields=["timestamp"], nearest=True, on="mouseover", empty=False)
    base = alt.Chart(df).encode(
        x=alt.X("timestamp:T", title="Date"),
        y=alt.Y("high:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max]))
    )
    line = base.mark_line(interpolate="monotone", color="#1f77b4")
    points = base.mark_circle(size=60, color="#1f77b4").encode(
        tooltip=[
            alt.Tooltip("timestamp:T", title="Timestamp"),
            alt.Tooltip("close:Q", title="Close", format=".2f"),
            alt.Tooltip("high:Q", title="High", format=".2f"),
            alt.Tooltip("low:Q", title="Low", format=".2f"),
            alt.Tooltip("volume:Q", title="Volume", format=".2f"),
            alt.Tooltip("marketCap:Q", title="Market Cap", format=".2f"),
        ],
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    ).add_params(hover)
    chart = (
        alt.layer(line, points)
        .properties(width="container", height=400)
        .configure_legend(orient="bottom")
    )
    with st.container(border=True):
        st.altair_chart(chart, use_container_width=True)

# ---------- Info Cards ----------
def show_info_cards(api_df):
    """Show financial predictor metrics in template style."""
    api_df['timestamp'] = pd.to_datetime(api_df['timestamp'])
    latest_time = api_df['timestamp'].max()
    df_24h = api_df[api_df['timestamp'] >= latest_time - timedelta(hours=24)]

    latest = api_df.iloc[-1]
    current_marketcap = latest['marketCap']
    current_volume = latest['volume']
    high_24h = df_24h['high'].max() if not df_24h.empty else latest['high']
    low_24h = df_24h['low'].min() if not df_24h.empty else latest['low']
    avg_open = api_df['open'].mean() if not api_df.empty else latest['open']
    avg_close = api_df['close'].mean() if not api_df.empty else latest['close']

    st.markdown("## Financial Predictors")
    cols = st.columns(6, gap="medium")

    metrics = [
        ("High (24h)", format_large_number(high_24h)),
        ("Low (24h)", format_large_number(low_24h)),
        ("Open", format_large_number(avg_open)),
        ("Close", format_large_number(avg_close)),
        ("Volume (24h)", format_large_number(current_volume)),
        ("Market Cap", format_large_number(current_marketcap))
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value)


def show_price_change_card(api_df):
    """Show 24h price change as template-style metric with delta."""
    api_df['timestamp'] = pd.to_datetime(api_df['timestamp'])
    latest_time = api_df['timestamp'].max()
    df_24h = api_df[api_df['timestamp'] >= latest_time - timedelta(hours=24)]

    latest = api_df.iloc[-1]
    current_close = latest['close']
    close_24h_ago = df_24h.iloc[0]['close'] if len(df_24h) > 1 else current_close
    pct_change_24h = ((current_close - close_24h_ago) / close_24h_ago) * 100 if close_24h_ago != 0 else 0

    st.metric(
        label="Price Change (24h)",
        value=f"{pct_change_24h:.2f}%",
        delta=f"{pct_change_24h:.2f}%"
    )


# ---------- Prediction Line Chart ----------
def plot_prediction_vs_actual(df_real, period_label, period_param):
    """Display predicted vs actual high price trend for the selected period."""
    st.markdown(f"### 📊 {period_label} Prediction vs Actual")
    df_real["timestamp"] = pd.to_datetime(df_real["timestamp"], format="%Y-%m-%d %H:%M:%S")
    with st.spinner(f"Fetching {period_label.lower()} prediction..."):
        data = fetch_prediction(period_param)
        if not data:
            st.warning("No data returned from API.")
            return
        if period_param == "1w" and "7d_prediction" in data:
            pred_list = data["7d_prediction"]
        elif period_param == "1d" and "next_day_prediciton" in data:
            pred_list = [data["next_day_prediciton"]]
        else:
            st.warning("API response missing prediction field.")
            st.write(data)
            return
        start_date = pd.to_datetime(data["start_date"])
        end_date = pd.to_datetime(data["end_date"])
        date_range = pd.date_range(start=start_date, end=end_date, periods=len(pred_list))
        df_pred = pd.DataFrame({"timestamp": date_range, "Predicted High": pred_list})
        df_real = df_real.copy()
        df_real["timestamp"] = pd.to_datetime(df_real["timestamp"]).dt.tz_localize(None)
        df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"]).dt.tz_localize(None)
        y_min = min(df_real["high"].min(), df_pred["Predicted High"].min())
        y_max = max(df_real["high"].max(), df_pred["Predicted High"].max())
        real_line = alt.Chart(df_real).mark_line(color="#1f77b4").encode(
            x=alt.X("timestamp:T", title="Date"),
            y=alt.Y("high:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max])),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("high:Q", title="Actual High", format=".2f"),
            ]
        )
        pred_line = alt.Chart(df_pred).mark_line(strokeDash=[5, 5], color="#ff7f0e").encode(
            x=alt.X("timestamp:T", title="Date"),
            y=alt.Y("Predicted High:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max])),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("Predicted High:Q", title="Predicted High", format=".2f"),
            ]
        )
        combined = (
            alt.layer(real_line, pred_line)
            .properties(width="container", height=400)
            .configure_legend(orient="bottom")
        )
        combined = (real_line + pred_line).properties(width="container", height=400).configure_legend(orient="bottom")
        with st.container(border=True):
            st.altair_chart(combined, use_container_width=True)
        # st.dataframe(df_pred, use_container_width=True)

def plot_next_day_prediction(df_real):
    """Display next-day high price prediction along with the last 7 days' real highs."""
    st.markdown("### 🌤️ Next-Day Prediction")
    with st.spinner("Fetching next-day prediction..."):
        data = fetch_prediction("1d")
        if not data or "next_day_prediciton" not in data:
            st.warning("API response missing next-day prediction field.")
            st.write(data)
            return
        next_day_high = data["next_day_prediciton"]
        df_real = df_real.copy()
        df_real["timestamp"] = pd.to_datetime(df_real["timestamp"])
        last_date = df_real["timestamp"].max()
        next_date = last_date + pd.Timedelta(days=1)
        df_real = df_real[df_real["timestamp"] >= last_date - pd.Timedelta(days=6)]
        df_pred = pd.DataFrame({
            "timestamp": [last_date, next_date],
            "Predicted High": [df_real["high"].iloc[-1], next_day_high]
        })
        y_min = min(df_real["high"].min(), next_day_high)
        y_max = max(df_real["high"].max(), next_day_high)
        real_line = alt.Chart(df_real).mark_line(color="#1f77b4").encode(
            x=alt.X("timestamp:T", title="Date"),
            y=alt.Y("high:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max])),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Date"),
                alt.Tooltip("high:Q", title="Actual High", format=".2f")
            ]
        )
        pred_line = alt.Chart(df_pred).mark_line(color="#ff7f0e", strokeDash=[5, 5]).encode(
            x=alt.X("timestamp:T", title="Date"),
            y=alt.Y("Predicted High:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max])),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Date"),
                alt.Tooltip("Predicted High:Q", title="Predicted High", format=".2f")
            ]
        )
        chart = (
            alt.layer(real_line, pred_line)
            .properties(width="container", height=400)
            .configure_legend(orient="bottom")
        )
        with st.container(border=True):
            st.altair_chart(chart, use_container_width=True)



# ---------- MAIN APP ----------
def run():
    st.title("Ethereum Price Prediction")
    st.write("Explore the latest trends, forecasts, and insights from the live prediction model")

    now = datetime.now(timezone.utc).astimezone()

    st.markdown(f"#### {now.date()} & {now.tzinfo}")

    df = fetch_data()
    if df is None or df.empty:
        return

    if not df.empty:
        df = preprocess_data(df)

        # Financial Predictors
        show_info_cards(df)

        # Historical Trends
        st.markdown("## Historical Trend")
        col1, col2 = st.columns([3, 1], gap="medium")
        with col1:
            df_filter = filter_by_range(df.copy())
            plot_high_price(df_filter)
        with col2:
            show_price_change_card(df)

    # Forcast
    st.markdown(f"## Forcast")
    st.write("*Disclaimer placeolder*")
    df_recent = df.tail(7)
    if not df_recent.empty:
        col1, col2 = st.columns(2)
        with col1:
            plot_prediction_vs_actual(df_recent, "1-Week", "1w")

        # Next-Day prediction (button and plot handled internally)
        with col2:
            plot_next_day_prediction(df_recent)

    # Dataframe
    st.dataframe(df[:10], use_container_width=True)
