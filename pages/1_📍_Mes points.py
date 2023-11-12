import streamlit as st
import pandas as pd

st.title("📍 Mes points")

data = pd.DataFrame(columns = ["Nom", "Longitude", "Latitude", "Image", "État"])

st.dataframe(data)


list_coords = []

def submit(nom, long, lat):
    list_coords.append((long,lat))

st.write("---")
st.write("# Ajouter un lieu de surveillance")
with st.form("Lieux", clear_on_submit=True):
    col1, col2 = st.columns(2)
    col = st.columns(1)[0]
    nom = col.text_input("Nom")
    long = col1.number_input("Latitude")
    lat = col2.number_input("Longitude")
    valid_status = st.form_submit_button("Valider", on_click=submit(nom, long, lat))
    if valid_status:
        st.success("Point ajouté")
        data.loc[len(data)] = [nom, long, lat, "", "Pas de fuite"]
        st.dataframe(data)