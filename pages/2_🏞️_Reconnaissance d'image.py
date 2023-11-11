import streamlit as st

st.title("🏞️ Reconnaissance d'image")

st.write("Importez une image satellite pour détecter la présence d'un puits de méthane")

upload = st.file_uploader(label = "Importez une image", type = ["png","jpg","jpeg","tif"])

if upload is not None:
    st.write("Importation réussie")