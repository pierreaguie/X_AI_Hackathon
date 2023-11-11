import streamlit as st
import pandas as pd

st.title("📍 Mes points")

data = pd.DataFrame(columns = ["Nom", "Longitude", "Latitude", "Image", "État"])

st.dataframe(data)