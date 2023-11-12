import streamlit as st

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


st.title("A propos de CleanR")

st.subheader("CleanR est une application web qui suit les fuites de méthane à travers le monde avec une IA révolutionnaire.")

st.markdown("---")

st.image("data/images/methane.jpg")

st.markdown("---")

st.markdown("## Qu'est-ce que le méthane? 🔥")

st.markdown("Méthane est un gaz à effet de serre qui est 84 fois plus puissant que le dioxyde de carbone. C'est le deuxième gaz à effet de serre le plus répandu émis par les activités humaines. Le méthane est émis par des sources naturelles telles que les zones humides, ainsi que par des activités humaines telles que les fuites des systèmes de gaz naturel et l'élevage de bétail.")

st.markdown("---")

st.markdown("## Qu'est-ce que CleanR? 🌬️")

st.markdown("CleanR est une application web qui suit les fuites de méthane à travers le monde avec une IA révolutionnaire. L'application est basée sur des données satellitaires prises en temps réel ou uploadée sur le site. L'application est développée par des étudiants de l'Ecole polytechnique.")

st.markdown("---")

st.markdown("## Comment utiliser CleanR? 🌍")

st.markdown("Pour utiliser CleanR, il suffit de rentrer les coordonnées des sites à surveiller sur la page 'mes points'.")

st.markdown("---")

st.markdown("## Qui sommes-nous? 👨‍💻")

st.markdown(" ")
st.markdown(" ")

st.image("data/images/grp.jpg", width=704)

st.markdown("Nous sommes des étudiants de l'Ecole polytechnique, nous avons développé cette application dans le cadre du du Hackathon de Quantum Black.")

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")

col1, col2 = st.columns(2)

col1.image("data/images/axel.jpg", width=300)
col1.subheader("Axel")
col1.markdown("**Chef de projet**")
col1.markdown("*Data Engineer*")
col1.markdown(" ")
col1.markdown(" ")
col1.markdown(" ")

col1.image("data/images/dim.jpg", width=300)
col1.subheader("Dimitri")
col1.markdown("**Chef de projet**")
col1.markdown("*Product designer*")


col2.image("data/images/dag.jpg", width=300)
col2.subheader("Pierre")
col2.markdown("**Chef de projet**")
col2.markdown("*Data Scientiist*")
col2.markdown(" ")
col2.markdown(" ")
col2.markdown(" ")

col2.image("data/images/lou.jpg", width=300)
col2.subheader("Louis")
col2.markdown("**Chef de projet**")
col2.markdown("*Sales manager*")
