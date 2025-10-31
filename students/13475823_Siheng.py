import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import datetime, timedelta


API_URL = "http://127.0.0.1:8000"
API_URL_PREDICT = f"{API_URL}/predict"


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


# ---------- DATA PREPROCESSING ----------
def preprocess_data(df):
    """Convert timestamp and sort values."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("timestamp")


def filter_by_range(df):
    """Allow user to filter dataframe by date range via dropdown."""
    st.markdown("### ⏳ Filter by Date Range")
    option = st.selectbox("Select Time Range:", ("All", "1 Week", "1 Month", "1 Year"), index=0)

    latest_date = df["timestamp"].max()
    if option == "1 Week":
        df = df[df["timestamp"] >= latest_date - timedelta(weeks=1)]
    elif option == "1 Month":
        df = df[df["timestamp"] >= latest_date - timedelta(days=30)]
    elif option == "1 Year":
        df = df[df["timestamp"] >= latest_date - timedelta(days=365)]

    return df


# ---------- CHARTS ----------
def plot_high_price(df):
    """Plot high price line chart with hover details."""
    y_min, y_max = df["high"].min(), df["high"].max()
    hover = alt.selection_point(fields=["timestamp"], nearest=True, on="mouseover", empty=False)

    base = alt.Chart(df).encode(
        x=alt.X("timestamp:T", title="Date"),
        y=alt.Y("high:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max])),
    )

    line = base.mark_line(interpolate="monotone")
    points = base.mark_circle(size=60).encode(
        tooltip=[
            alt.Tooltip("timestamp:T", title="Timestamp"),
            alt.Tooltip("close:Q", title="Close", format=".2f"),
            alt.Tooltip("high:Q", title="High", format=".2f"),
            alt.Tooltip("low:Q", title="Low", format=".2f"),
            alt.Tooltip("volume:Q", title="Volume", format=".2f"),
            alt.Tooltip("marketCap:Q", title="Market Cap", format=".2f"),
        ],
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    ).add_params(hover)

    chart = (line + points).properties(width="container", height=400)
    st.subheader("High Price")
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

# ---------- PREDICTION ----------
def fetch_prediction(period):
    """Fetch prediction data from API with a given period."""
    try:
        response = requests.post(API_URL_PREDICT, params={"period": period}, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling prediction API: {e}")
        return None

def plot_prediction_vs_actual(df_real, period_label, period_param):
    """Plot predicted vs real high price trend for a given period."""
    df_real["timestamp"] = pd.to_datetime(df_real["timestamp"], format="%Y-%m-%d %H:%M:%S")

    st.subheader(f"{period_label} Prediction vs Actual")

    with st.spinner(f"Fetching {period_label.lower()} prediction..."):
        data = fetch_prediction(period_param)
        if not data:
            st.warning("No data returned from API.")
            return

        # handle prediction field dynamically
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

        real_line = (
            alt.Chart(df_real)
            .mark_line(color="steelblue")
            .encode(
                x=alt.X("timestamp:T", title="Date"),
                y=alt.Y("high:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max])),
                tooltip=["timestamp", "high"]
            )
        )

        pred_line = (
            alt.Chart(df_pred)
            .mark_line(strokeDash=[5, 5], color="orange")
            .encode(
                x=alt.X("timestamp:T", title="Date"),
                y=alt.Y("Predicted High:Q", title="High Price", scale=alt.Scale(domain=[y_min, y_max])),
                tooltip=["timestamp", "Predicted High"]
            )
        )

        combined = (real_line + pred_line).properties(width="container", height=400)
        st.altair_chart(combined, use_container_width=True)
        st.dataframe(df_pred, use_container_width=True)

def plot_next_day_prediction(df_real):
    """Fetch next-day prediction from API and plot last 7 days real high and next-day prediction."""
    st.subheader("Next-Day Prediction vs Last 7 Days Actual")

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

        # DataFrame for the dotted prediction line
        df_pred = pd.DataFrame({
            "timestamp": [last_date, next_date],
            "Predicted High": [df_real["high"].iloc[-1], next_day_high]
        })

        y_min = min(df_real["high"].min(), next_day_high)
        y_max = max(df_real["high"].max(), next_day_high)

        # Solid line for real 7 days
        real_line = (
            alt.Chart(df_real)
            .mark_line(color="steelblue")
            .encode(
                x="timestamp:T",
                y=alt.Y("high:Q", scale=alt.Scale(domain=[y_min, y_max])),
                tooltip=["timestamp", "high"]
            )
        )

        # Dotted line for next-day prediction
        pred_line = (
            alt.Chart(df_pred)
            .mark_line(color="orange", strokeDash=[5, 5])
            .encode(
                x="timestamp:T",
                y=alt.Y("Predicted High:Q", scale=alt.Scale(domain=[y_min, y_max])),
                tooltip=["timestamp", "Predicted High"]
            )
        )

        chart = (real_line + pred_line).properties(width="container", height=400)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df_pred, use_container_width=True)

# ---------- MAIN APP ----------
def run():
    st.title("AT3 Advanced Machine Learning Model 2")

    df = fetch_data()
    if df is None or df.empty:
        return

    df = preprocess_data(df)
    df = filter_by_range(df)

    plot_high_price(df)
    plot_market_cap(df)

    # Prediction section
    st.subheader("Predictions")

    # Plot charts full width based on button click
    df_recent = df.tail(7)
    if not df_recent.empty:
        tab1, tab2 = st.tabs(["1-Week Prediction", "Next-Day Prediction"])

        with tab1:
            plot_prediction_vs_actual(df_recent, "1-Week", "1w")
        
        with tab2:
            plot_next_day_prediction(df_recent)



if __name__ == "__main__":
    run()
