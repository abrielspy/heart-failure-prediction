import streamlit as st
from function import load_data, predict
from Tabs import home, predict

Tabs = {
    "Home": home,
    "Prediction": predict
}

st.sidebar.title("Navigation")

selection = st.sidebar.radio("Go to", list(Tabs.keys()))

df, x, y = load_data()

if selection in ["Prediction"]:
  Tabs[selection].app(df, x, y)
else:
  Tabs[selection].app()