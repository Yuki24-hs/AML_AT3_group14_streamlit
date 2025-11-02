# app/main.py
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="AT3 Group 14", layout="wide", initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Map visible page names -> module import paths
MODULE_MAP = {
    "Bitcoin": "students.25229384_Yukthi",
    "Ethereum": "students.13475823_Siheng",
    "XRP": "students.25658663_Queenie",
    "Solana": "students.25428006_Mitali",
}

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
    - Member 1: Mitali H Balki (25428006) 
    - Member 2: Siheng Li (XXX)
    - Member 3: Queenie Goh (XXX)
    - Member 4: Yukthi Hosadurga Shivalingegowda (25229384)

    **Next steps / risks:** Later.
    """
    )

st.sidebar.title("Navigation")
nav_options = ["Main"] + list(MODULE_MAP.keys())
page = st.sidebar.radio("Go to", nav_options, label_visibility="collapsed")

if page == "Main":
    show_main()
else:
    module_path = MODULE_MAP.get(page)
    if not module_path:
        st.error(f"Page {page!r} is not configured. Available: {', '.join(MODULE_MAP)}")
        st.stop()

    try:
        mod = importlib.import_module(module_path)
        if hasattr(mod, "run") and callable(mod.run):
            mod.run()
        else:
            st.error(f"`{module_path}.run()` not found. Add a `run()` function.")
    except ModuleNotFoundError as e:
        st.error(f"Module not found: {module_path}\n\nDetails: {e}")
