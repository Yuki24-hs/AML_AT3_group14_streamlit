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


# ---------- Function & Helper ----------
def preprocess_data(df):
    """Convert timestamp and sort values."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("timestamp")

def format_large_number(num):
    """Format large numbers with units M (million) or B (billion)."""
    if abs(num) >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    else:
        return f"${num:,.0f}"


# ---------- Display Plots ----------
def fetch_prediction(period):
    """Fetch prediction data from API with a given period."""
    try:
        response = requests.post(API_URL_PREDICT, params={"period": period}, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling prediction API: {e}")
        return None

def show_info_cards(api_df):
    """Display crypto metrics inside bordered cards."""
    api_df['timestamp'] = pd.to_datetime(api_df['timestamp'])
    latest_time = api_df['timestamp'].max()

    df_1h = api_df[api_df['timestamp'] >= latest_time - timedelta(hours=1)]
    df_24h = api_df[api_df['timestamp'] >= latest_time - timedelta(hours=24)]

    latest = api_df.iloc[-1]
    current_close = latest['close']
    current_marketcap = latest['marketCap']
    current_volume = latest['volume']

    close_1h_ago = df_1h.iloc[0]['close'] if len(df_1h) > 1 else current_close
    close_24h_ago = df_24h.iloc[0]['close'] if len(df_24h) > 1 else current_close

    pct_change_1h = ((current_close - close_1h_ago) / close_1h_ago) * 100 if close_1h_ago != 0 else 0
    pct_change_24h = ((current_close - close_24h_ago) / close_24h_ago) * 100 if close_24h_ago != 0 else 0

    high_24h = df_24h['high'].max() if not df_24h.empty else latest['high']
    low_24h = df_24h['low'].min() if not df_24h.empty else latest['low']
    vol_mkt_ratio = current_volume / current_marketcap if current_marketcap != 0 else 0

    st.markdown("""
        <style>
            .metric-card {
                border: 2px solid #4CAF50;
                border-radius: 12px;
                padding: 16px;
                background-color: #f9f9f9;
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 15px;
            }
            .metric-label {
                font-size: 18px;
                color: #111;
                font-weight: 800;
                margin-bottom: 6px;
            }
            .metric-value {
                font-size: 22px;
                font-weight: 700;
                color: #222;
            }
            .metric-delta {
                font-size: 14px;
                color: #008000;
                font-weight: 600;
            }
            .metric-delta.negative {
                color: #cc0000;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Market Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Cap</div>
                <div class="metric-value">{format_large_number(current_marketcap)}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volume (24h)</div>
                <div class="metric-value">{format_large_number(current_volume)}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">High (24h)</div>
                <div class="metric-value">${high_24h:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Low (24h)</div>
                <div class="metric-value">${low_24h:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        delta_1h_class = "metric-delta" if pct_change_1h >= 0 else "metric-delta negative"
        delta_24h_class = "metric-delta" if pct_change_24h >= 0 else "metric-delta negative"

        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Price Change (1h)</div>
                <div class="metric-value">{pct_change_1h:.2f}%</div>
                <div class="{delta_1h_class}">{'▲' if pct_change_1h >= 0 else '▼'} {abs(pct_change_1h):.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Price Change (24h)</div>
                <div class="metric-value">{pct_change_24h:.2f}%</div>
                <div class="{delta_24h_class}">{'▲' if pct_change_24h >= 0 else '▼'} {abs(pct_change_24h):.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Vol/Mkt Cap (24h):** {vol_mkt_ratio:.4f}")


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
        # st.dataframe(df_pred, use_container_width=True)

def plot_next_day_prediction(df_real):
    """Fetch next-day prediction from API and plot last 7 days real high and next-day prediction."""
    st.subheader("Next-Day Prediction")

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
        # st.dataframe(df_pred, use_container_width=True)

# ---------- MAIN APP ----------
def run():
    st.title("Ethereum(ETH) Price Prediction")
    st.write("*Description placeolder*")

    df = fetch_data()
    if df is None or df.empty:
        return

    df = preprocess_data(df)

    # Market Overview
    show_info_cards(df)

    # Plot charts full width based on button click
    df_recent = df.tail(7)
    if not df_recent.empty:
        col1, col2 = st.columns(2)
        with col1:
            plot_prediction_vs_actual(df_recent, "1-Week", "1w")

        # Next-Day prediction (button and plot handled internally)
        with col2:
            plot_next_day_prediction(df_recent)

        # tab1, tab2 = st.tabs(["1-Week Prediction", "Next-Day Prediction"])

        # with tab1:
        #     plot_prediction_vs_actual(df_recent, "1-Week", "1w")
        
        # with tab2:
        #     plot_next_day_prediction(df_recent)



# if __name__ == "__main__":
#     run()
