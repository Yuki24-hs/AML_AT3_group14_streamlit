import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import datetime, timedelta

API_URL = "https://at3-api-25229384-latest.onrender.com"
API_URL_PREDICT = f"{API_URL}/predict/bitcoin"
API_URL_7d = f"{API_URL}/predict/bitcoin/7d"
















def run():
    st.title("AT3 Advanced Machine Learning Model 4")
    
    
