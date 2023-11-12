import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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


st.title("🏞️ Reconnaissance d'image")

st.write("Importez une image satellite pour détecter la présence d'un puits de méthane")

model = tf.keras.models.load_model("/Users/pierreaguie/X_AI_Hackathon/models/310.h5")

upload = st.file_uploader(label = "Importez une image", type = ["png","jpg","jpeg","tif"])

st.set_option('deprecation.showPyplotGlobalUse', False)




if upload is not None:    
    img = load_img(upload, target_size = (64,64), color_mode = "grayscale")
    


    img_array = img_to_array(img)/65535

    input_data = img_array.reshape(1, 64, 64, 1).astype('float32')

    pred = model.predict(input_data)
    pred_argmax = pred.argmax()
    if pred_argmax == 0:
        st.write("Cette image ne présente pas de fuite")
    else:
        st.write("Cette image présente une fuite")
    

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    st.pyplot()

