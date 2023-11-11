import streamlit as st

st.set_page_config(
    page_title="CleanR",
    page_icon="🌍",
)

st.write("# CleanR 🌍")

st.write("""
    CleanR est l'outil IA destiné à la détection de fuites de méthane à partir d'images satellite.
""")

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)