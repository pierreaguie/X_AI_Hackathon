import streamlit as st
import pandas as pd

st.title("📍 Mes points")

if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns = ["Nom", "Longitude", "Latitude", "Image", "État"])




def submit(nom, long, lat):
    n = len(st.session_state['data'])
    st.session_state['data'].loc[n] = [nom, long, lat, "", "Pas de fuite"]

st.write("---")
st.write("# Ajouter un lieu de surveillance")
with st.form("Lieux", clear_on_submit=True):
    col1, col2 = st.columns(2)
    col = st.columns(1)[0]
    nom = col.text_input("Nom")
    long = col1.number_input("Latitude", value=0.0, min_value=-90.0, max_value=90.0)
    lat = col2.number_input("Longitude", value=0.0, min_value=-180.0, max_value=180.0)
    valid_status = st.form_submit_button("Valider")
    if valid_status:
        if nom == "":
            st.warning("Veuillez renseigner un nom")
        else:
            n = len(st.session_state['data'])
            st.session_state['data'].loc[n] = [nom, long, lat, "", "Pas de fuite"]
            st.success("Point ajouté")

st.markdown(" ")
st.markdown("---")
st.markdown(" ")
st.write("# Mes points")
st.dataframe(st.session_state['data'], width = 704)


st.markdown("""
<style>
        .st-emotion-cache-zq5wmm.ezrtsby0
        {
            visibility: hidden;
        }
        .st-emotion-cache-cio0dv.ea3mdgi1
        {
            visibility: hidden;
        }
""", unsafe_allow_html=True)
