import streamlit as st
from predict_page import show_predict_page
from models_page import show_models_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Features", ("Predict", "Explore", "Models"))
if page == "Explore":
  show_explore_page()
elif page == "Models":
  show_models_page()
else:
  show_predict_page()
