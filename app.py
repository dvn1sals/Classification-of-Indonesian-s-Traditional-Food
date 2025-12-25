import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# CONFIG

IMG_SIZE = (160, 160)

st.set_page_config(
    page_title="Klasifikasi Makanan Tradisional Indonesia",
    layout="centered"
)

st.title("🍽️ Klasifikasi Makanan Tradisional Indonesia")
st.write("Upload gambar makanan dan pilih model klasifikasi")


# LOAD MODEL (CACHED)

@st.cache_resource
def load_model(model_name):
    if model_name == "CNN Base":
        return tf.keras.models.load_model("Model\cnn_base.h5")
    elif model_name == "MobileNetV2":
        return tf.keras.models.load_model("Model\mobilenet.h5")
    elif model_name == "ResNet50":
        return tf.keras.models.load_model("Model\\resnet.h5")


# PILIH MODEL

model_choice = st.selectbox(
    "Pilih Model",
    ["CNN Base", "MobileNetV2", "ResNet50"]
)

model = load_model(model_choice)

# CLASS NAMES

class_names = ["Gado-Gado","Nasi Goreng", "Nasi Padang","Rendang","Sate"]  


# UPLOAD IMAGE
uploaded_file = st.file_uploader(
    "Upload Gambar Makanan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", width=400)


    
    # PREPROCESS
    
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)


    
    # PREDICT
    
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    
    # OUTPUt
    
    st.subheader("Hasil Prediksi")
    st.success(f"🍴 **{class_names[class_idx]}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")

    # Optional: tampilkan semua probabilitas
    with st.expander("Lihat Probabilitas Semua Kelas"):
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {pred[0][i]*100:.2f}%")
