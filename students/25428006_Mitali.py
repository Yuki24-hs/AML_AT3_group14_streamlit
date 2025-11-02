import streamlit as st
import requests
import urllib
import pandas as pd
import json
from datetime import datetime,  timedelta
from requests.exceptions import RequestException
import time
import altair as alt

# define variables --------------------------------------------------------------------------------
API_URL = "https://at3-api-25428006.onrender.com/"
API_HEALTH_POINT = 'health/'
API_PREDICT_ND = 'predict/solana/'
API_PREDICT_7D = 'predict/solana/7d'
API_DATE = 'get_date/'

# for kraken
pair = "SOL/USD"
interval = 1440

# CSV path
csv_path = 'data/25428006-AT3-Solana.csv'

# helper Functions --------------------------------------------------------------------------------
# define fucntions to get data
# get data from kraken (by default fetches last 720 days)
def get_kraken_data(pair, interval):
   response = request(
      method="GET",
      path="/0/public/OHLC",
      query={
         "pair": pair,
         "interval": interval,
      },
      environment="https://api.kraken.com",
   )
   return response.read().decode()

def request(method: str = "GET", 
            path: str = "", 
            query: dict | None = None, 
            body: dict | None = None, 
            environment: str = ""):
   url = environment + path
   query_str = ""
   if query is not None and len(query) > 0:
      query_str = urllib.parse.urlencode(query)
      url += "?" + query_str

   req = urllib.request.Request(
      method=method,
      url=url
   )
   return urllib.request.urlopen(req)

# function to fetch data 
def fetch_data(pair, interval):
   data = get_kraken_data(pair, interval)
   json_data = json.loads(data)
   result = json_data["result"]
   df_api = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'], 
                         data = result['SOL/USD'])
   return df_api

# function to get data from kraken API
def get_api_data():
  """
  Get data for the last 720 days from Kraken.
  """
  pair = "SOL/USD"
  interval = 1440

  # fetch data
  api_data = fetch_data(pair, interval)

  # define numerical columns
  numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']

  # make numerical columns into numeric type
  api_data[numeric_cols] = api_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

  # calculate marketCap
  api_data['marketCap'] = api_data['vwap']*api_data['volume']

  # convert timestamp to date
  api_data['date'] = pd.to_datetime(api_data['timestamp'], unit='s', utc=True)

  return api_data[['date', 'open', 'high', 'low', 'close', 'volume', 'marketCap']]

# function to get data from csv
def get_csv_data(file_path):
  """
  Get data from a CSV file and return a DataFrame.
  """
  # get data from csv file
  csv_data = pd.read_csv(file_path)

  # convert timeopen to datetime
  csv_data['date'] = pd.to_datetime(csv_data['timeOpen'])

  # keep only required columns
  csv_data = csv_data[['date', 'open', 'high', 'low', 'close', 'volume', 'marketCap']]

  # sort data and reset index
  csv_data = csv_data.sort_values(by='date')
  csv_data = csv_data.reset_index(drop=True)

  return csv_data

# function to combine the dataframes
# get and cache data
@st.cache_data(ttl=3600)  # cache for 1 hour
def combine_data(file_path):
  """
  Combine the data from the API and CSV files.
  """
  # get data from api
  api_data = get_api_data()
  csv_data = get_csv_data(file_path)

  # get the min date from API data
  min_date = api_data['date'].min()

  # filter the CSV data to only include dates before the min_date
  csv_data = csv_data[csv_data['date'] < min_date]

  # concatenate datasets
  final_data = pd.concat([api_data[['date', 'open', 'high', 'low', 'close', 'volume', 'marketCap']], csv_data])

  # Sort by date and reset index
  final_data = final_data.sort_values(by='date').reset_index(drop=True)

  return final_data

# @st.cache_data(ttl=3600) # cache predicted data for an hour
# def get_7d_data(url_path):
#   """
#   Gets the predicted data for the past 7 days (including today)
#   """
#   # get data from predicted 7days
#   response = requests.get(url_path)

#   # Check if the request was successful
#   if response.status_code == 200:
#       api_7d_data = response.json()  # Parse JSON response
#   else:
#       return f"Failed to fetch data. Status code: {response.status_code}"
  
#   # convert data to dataframe
#   pred_dates = api_7d_data['prediction']['pred_date']
#   pred_prices = api_7d_data['prediction']['price']
#   api_7d_data = pd.DataFrame(data={'date': pred_dates, 'predicted_high': pred_prices})

#   # sort dataframe
#   api_7d_data['date'] = pd.to_datetime(api_7d_data['date'])

#   return api_7d_data

@st.cache_data(ttl=3600)  # cache predicted data for an hour
def get_7d_data(url_path):
    """
    Safely fetches predicted data for the past 7 days (including today)
    with retries and graceful error handling.
    """
    max_retries = 3
    backoff = 20  # seconds between retries

    for attempt in range(max_retries):
        try:
            response = requests.get(url_path, timeout=800)  # ⏱ timeout in seconds

            # Check if request was successful
            if response.status_code == 200:
                api_7d_data = response.json()
                pred_dates = api_7d_data['prediction']['pred_date']
                pred_prices = api_7d_data['prediction']['price']

                df = pd.DataFrame({'date': pred_dates, 'predicted_high': pred_prices})
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date').reset_index(drop=True)
                return df

            else:
                st.warning(f"⚠️ Failed to fetch data (Status {response.status_code}). Retrying...")
        
        except RequestException as e:
            st.warning(f"⚠️ Attempt {attempt+1}/{max_retries} failed: {e}")
        
        time.sleep(backoff)  # wait before retrying

    # If all retries fail, return a safe empty DataFrame or message
    st.error("🚫 Could not retrieve prediction data after multiple attempts. Please try again later.")
    return pd.DataFrame(columns=['date', 'predicted_high'])

@st.cache_data(ttl=3600)  # cache for 5 minutes (adjust as needed)
def get_current_date(url):
    """Fetch current date from API and cache the result."""
    response = requests.get(url, timeout=800)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch date. Status code: {response.status_code}")
        return None

def format_millions(value):
    """Convert large numbers to human-readable format (millions)."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return str(value)

# -----------------------------------------------------------------------------------------------

# get the cached combined data
combined_data = combine_data(csv_path)

# Get the last two days
current_high = combined_data.iloc[-1]['high']
previous_high = combined_data.iloc[-2]['high']

current_low = combined_data.iloc[-1]['low']
previous_low = combined_data.iloc[-2]['low']

current_open = combined_data.iloc[-1]['open']
previous_open = combined_data.iloc[-2]['open']

current_close = combined_data.iloc[-1]['close']
previous_close = combined_data.iloc[-2]['close']

current_volume = combined_data.iloc[-1]['volume']
previous_volume = combined_data.iloc[-2]['volume']

current_marketCap = combined_data.iloc[-1]['marketCap']
previous_marketCap = combined_data.iloc[-2]['marketCap']

# Running the app --------------------------------------------------------------------------------

def run():
    st.title("SOLANA SOL Dashboard")

    st.markdown(
    """
    ## 🌐 About Solana (SOL)
    **Solana (SOL)** is a high-performance blockchain platform launched in **2020** by **Anatoly Yakovenko**, designed for **scalable**, **low-cost**, and **lightning-fast** decentralized transactions. Powered by its unique **Proof-of-History (PoH)** mechanism, Solana enables developers to build **DeFi**, **NFT**, and **high-frequency trading** applications, capable of processing **thousands of transactions per second** with minimal fees, making it one of the fastest and most efficient blockchain networks today.  
    """
    )

    with st.spinner("🔄 Loading date..."):
        api_date = get_current_date(API_URL + API_DATE)
        if api_date and "current_date" in api_date:
            current_date_str = api_date["current_date"]

            # Convert to datetime object
            current_date = datetime.fromisoformat(current_date_str)

            # Format as YYYY-MM-DD HH:MM
            formatted_date = current_date.strftime("%Y-%m-%d %H:%M")

            # Display in Streamlit
            st.markdown(f"**🕒 Current Time Zone for Data:** {formatted_date}")
        else:
            st.warning("⚠️ Could not load current date from API.")
    
    st.markdown(
    """
    ## Financial Predictors
    """
    )

    with st.container():
        cols = st.columns(6, gap="medium")

        with cols[0]:
            st.metric(
                "High",
                f"{current_high:0.1f}",
                delta=f"{(((current_high - previous_high)/previous_high)*100):0.2f} %"
            )
        
        with cols[1]:
            st.metric(
                "Low",
                f"{current_low:0.1f}",
                delta=f"{(((current_low - previous_low)/previous_low)*100):0.2f} %"
            )
        
        with cols[2]:
            st.metric(
                "Open",
                f"{current_open:0.1f}",
                delta=f"{(((current_open - previous_open)/previous_open)*100):0.2f} %"
            )
        
        with cols[3]:
            st.metric(
                "Close",
                f"{current_close:0.1f}",
                delta=f"{(((current_close - previous_close)/previous_close)*100):0.2f} %"
            )
        
        with cols[4]:
            st.metric(
                "Volume",
                f"{current_volume:0.1f}",
                delta=f"{(((current_volume - previous_volume)/previous_volume)*100):0.2f} %"
            )
        
        with cols[5]:
            st.metric(
                "Market Capital",
                f"{format_millions(current_marketCap)}",
                delta=f"{(((current_marketCap - previous_marketCap)/previous_marketCap)*100):0.2f} %"
            )
        
        cols2 = st.columns([3,1], gap="medium")

        with cols2[0]:
            TIMEFRAME = ['1W', '6M', '1Y', 'Max']
            tabs = st.tabs(TIMEFRAME)

            today = combined_data['date'].max()

            tab_ranges = {
                "1W": today - timedelta(days=7),
                "6M": today - timedelta(days=182),  # ~6 months
                "1Y": today - timedelta(days=365),
                "Max": combined_data['date'].min()
            }

            for tab_name, tab in zip(tab_ranges.keys(), tabs):
                with tab:
                    start_date = tab_ranges[tab_name]
                    filtered_df = combined_data[combined_data['date'] >= start_date]

                    if filtered_df.empty:
                        st.warning("No data available for this period.")
                    else:
                        chart = alt.Chart(filtered_df).mark_line(point=False).encode(
                            x='date:T',
                            y='close:Q',  # You can choose close or high/low etc.
                            tooltip=[
                                alt.Tooltip('date:T', title='Date'),
                                alt.Tooltip('open:Q', title='Open', format=".2f", formatType="number"),
                                alt.Tooltip('high:Q', title='High', format=".2f", formatType="number"),
                                alt.Tooltip('low:Q', title='Low', format=".2f", formatType="number"),
                                alt.Tooltip('close:Q', title='Close', format=".2f", formatType="number"),
                                # alt.Tooltip('marketCap:Q', title='Market Cap', format=".2f", formatType="number")
                            ]
                        ).interactive()  # zoom & pan

                        st.altair_chart(chart, use_container_width=True)
        with cols2[1]:
            # st.metric(
            #     label="High Price Change 24H",
            #     value=f"${current_high - previous_high:,.2f}",
            #     delta=f"{(((current_high - previous_high)/previous_high)*100):0.2f}%"
            # )   
            pc_color = "green" if (current_high - previous_high) >= 0 else "red"
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    height: 400px;  /* adjust as needed */
                    width: 100%;
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    padding: 20px;
                ">
                    <div style="font-size: 18px; font-weight: 500;">High Price Change</div>
                    <div style="font-size: 18px; font-weight: 500;">24H</div>
                    <div style="font-size: 32px; font-weight: 700; margin-top: 10px; color: {pc_color};">${current_high - previous_high:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )  

    st.divider()
    
    # section for forecasts  
    st.markdown(
    """
    ## Financial Forecasts
    """
    )

    # get the cached predicted data
    with st.spinner("🔄 Loading predictions..."):
        predicted_data = get_7d_data(API_URL + API_PREDICT_7D)
        st.write('data was loaded')


