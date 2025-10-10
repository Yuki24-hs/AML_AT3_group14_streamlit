# app/main.py
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import streamlit as st

# page config
st.set_page_config(
    page_title="AT3 Group 14", layout="wide", initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent  # AT3_GROUP14/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Main", "Student 1", "Student 2", "Student 3", "Student 4"],
    label_visibility="collapsed",
)

MODULE_MAP = {
    "Student 1": "students.25428006_Mitali",
    "Student 2": "students.XXX_Siheng",
    "Student 3": "students.XXX_Queenie",
    "Student 4": "students.25229384_Yukthi",
}


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
        - Member 2: Siheng Li ( XXX)
        - Member 3: Queenie Goh ( XXX )
        - Member 4: Yukthi Hosadurga Shivalingegowda ( 25229384 )

        **Next steps / risks:** Later.
        """
    )


if page == "Main":
    show_main()
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
