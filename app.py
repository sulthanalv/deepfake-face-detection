import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🧠",
    layout="centered"
)

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenet_improvement.keras"
    )

model = load_model()

# ==========================
# Header
# ==========================
st.title("🧠 Deepfake Face Detection")

st.write(
    "Aplikasi deteksi citra deepfake wajah menggunakan "
    "**MobileNet** dengan metode **Transfer Learning**."
)

# ==========================
# Upload Image
# ==========================
uploaded_file = st.file_uploader(
    "Upload gambar wajah",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# Prediction
# ==========================
if uploaded_file is not None:

    # Membaca gambar
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    # Resize
    img = image.resize((128, 128))

    # Konversi ke array
    img = np.array(img, dtype=np.float32)

    # Normalisasi
    img = img / 255.0

    # Tambah dimensi batch
    img = np.expand_dims(img, axis=0)

    with st.spinner("Sedang melakukan prediksi..."):

        try:
            pred = model.predict(img, verbose=0)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar:\n\n{e}")
            st.stop()

    confidence = float(pred[0][0])

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    if confidence > 0.5:

        st.success("✅ REAL")

        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )

        st.info(
            "Model memprediksi gambar ini sebagai **REAL** "
            "berdasarkan fitur visual yang dipelajari selama proses training."
        )

    else:

        fake_conf = (1-confidence)*100

        st.error("❌ FAKE")

        st.metric(
            label="Confidence",
            value=f"{fake_conf:.2f}%"
        )

        st.warning(
            "Model mendeteksi adanya indikasi **manipulasi wajah (Deepfake)** "
            "berdasarkan fitur visual hasil ekstraksi."
        )

# ==========================
# Footer
# ==========================
st.markdown("---")

st.caption(
    "Skripsi - Perbandingan Model Pretrained Deep Learning "
    "MobileNet, VGG16, ResNet50 dan XceptionNet "
    "untuk Deteksi Citra Deepfake pada Wajah"
)