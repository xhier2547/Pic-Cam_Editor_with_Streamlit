import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

st.title("📷 Image Processing with Streamlit + Color Oscilloscope")

# --- Input source ---
option = st.radio("เลือกแหล่งที่มาของภาพ", ("Webcam", "Upload Image", "Image URL"))

# --- Get Image ---
image = None

if option == "Webcam":
    img_file_buffer = st.camera_input("ถ่ายภาพจาก Webcam")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)

elif option == "Upload Image":
    uploaded_file = st.file_uploader("อัปโหลดไฟล์รูปภาพ", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == "Image URL":
    url = st.text_input("ใส่ URL ของรูปภาพ:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("ไม่สามารถโหลดรูปจาก URL ได้")

# --- Processing ---
if image:
    st.image(image, caption="ต้นฉบับ", use_column_width=True)

    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_array

    st.subheader("🔧 Image Processing")

    process_type = st.selectbox("เลือกการประมวลผล", ("Grayscale", "Canny Edge Detection"))

    if process_type == "Grayscale":
        processed = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif process_type == "Canny Edge Detection":
        t1 = st.slider("Threshold1", 0, 255, 100)
        t2 = st.slider("Threshold2", 0, 255, 200)
        processed = cv2.Canny(img_cv, t1, t2)

    st.image(processed, caption="ผลการประมวลผล", use_column_width=True, channels="GRAY")

    # --- ปุ่ม Color Oscilloscope ---
    st.subheader("📊 Visualization")
    show_scope = st.checkbox("แสดง Color Oscilloscope")

    if show_scope:
        fig, ax = plt.subplots()
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img_cv], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
        ax.set_title("Color Oscilloscope (RGB Histogram)")
        ax.set_xlim([0, 256])
        st.pyplot(fig)
