import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenet_improvement.keras"
    )

model = load_model()

st.title("🧠 Deepfake Face Detection")
st.write(
    "Aplikasi deteksi citra deepfake wajah menggunakan "
    "model MobileNet dengan metode Transfer Learning."
)

uploaded_file = st.file_uploader(
    "Upload gambar wajah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

st.image(
    image,
    caption="Gambar yang diupload",
    use_container_width=True
)

img = image.resize((128, 128))
img = np.array(img, dtype=np.float32)

img = img / 255.0
img = np.expand_dims(img, axis=0)

try:
    pred = model.predict(img, verbose=0)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
    st.stop()

confidence = float(pred[0][0])

st.subheader("Hasil Prediksi")

if confidence > 0.5:
    st.success(f"REAL ({confidence*100:.2f}%)")
else:
    st.error(f"FAKE ({(1-confidence)*100:.2f}%)")

st.markdown("---")
st.caption(
    "Skripsi Deteksi Deepfake Wajah Menggunakan "
    "Transfer Learning dan MobileNet"
)