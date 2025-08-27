import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from streamlit_cropper import st_cropper

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Photoshop-lite App", layout="wide")
st.title("🖼 Photoshop-lite Image Editor with Layers")

# ---------------- INPUT ---------------- #
st.sidebar.header("📂 เลือกแหล่งภาพ")
option = st.sidebar.radio("เลือก input", ("Upload Image", "Image URL", "Webcam"))

image = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("อัปโหลดไฟล์รูปภาพหลัก", type=["jpg","png","jpeg"], key="base")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Image URL":
    url = st.text_input("ใส่ URL ของรูปภาพหลัก:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("โหลดรูปจาก URL ไม่สำเร็จ")

elif option == "Webcam":
    cam = st.camera_input("ถ่ายภาพจาก Webcam")
    if cam:
        image = Image.open(cam).convert("RGB")

# ---------------- SESSION STATE ---------------- #
if "layers" not in st.session_state:
    st.session_state.layers = []  # เก็บเลเยอร์ทั้งหมด

# ---------------- PROCESSING ---------------- #
if image:
    processed = np.array(image)

    # Layout: Tools | Preview | Graphs + Layers
    col1, col2, col3 = st.columns([1,2.5,1.5])

    # -------- Tools (ซ้าย) -------- #
    with col1:
        st.markdown("### 🛠 Tools")
        apply_crop   = st.checkbox("✂️ Crop")
        apply_gray   = st.checkbox("⚫ Grayscale")
        apply_canny  = st.checkbox("📏 Canny")
        apply_rotate = st.checkbox("🔄 Rotate")
        apply_detect = st.checkbox("🔎 Detect Face")

    # -------- Preview (กลาง) -------- #
    with col2:
        st.markdown("### 📌 Preview")

        img_cv = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        if apply_crop:
            st.info("ใช้เมาส์ลากเลือกพื้นที่ Crop")
            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color="#FF0000",
                aspect_ratio=None
            )
            processed = np.array(cropped_img)

        if apply_gray:
            processed = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        if apply_canny:
            t1 = st.slider("Threshold1", 0, 255, 100)
            t2 = st.slider("Threshold2", 0, 255, 200)
            processed = cv2.Canny(img_cv, t1, t2)

        if apply_rotate:
            angle = st.slider("หมุน (degree)", -180, 180, 0)
            h, w = processed.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            processed = cv2.warpAffine(img_cv, M, (w, h))

        if apply_detect:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            processed = img_cv

        st.image(
            processed,
            caption="ผลลัพธ์",
            channels="RGB" if processed.ndim == 3 else "GRAY",
            use_column_width=True
        )

    # -------- Graph + Layers (ขวา) -------- #
    with col3:
        st.markdown("### 📊 Graphs")
        if st.button("แสดง Histogram"):
            fig, ax = plt.subplots(figsize=(4,3))
            if processed.ndim == 3:
                colors = ('r','g','b')
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col)
            else:
                ax.hist(processed.ravel(), bins=256, color='gray')
            st.pyplot(fig)

        # -------- Layers -------- #
        st.markdown("### 🗂 Layers Panel")

        # init background layer
        if not any(l["name"] == "Background" for l in st.session_state.layers):
            st.session_state.layers.append({
                "id": len(st.session_state.layers),
                "name": "Background",
                "image": np.array(image),
                "visible": True,
                "opacity": 1.0
            })

        # add overlay
        overlay_file = st.file_uploader("เพิ่มเลเยอร์ใหม่ (Overlay)", type=["jpg","png","jpeg"], key=f"overlay_{len(st.session_state.layers)}")
        if overlay_file:
            overlay_img = Image.open(overlay_file).convert("RGB")
            overlay_img = overlay_img.resize((processed.shape[1], processed.shape[0]))
            st.session_state.layers.append({
                "id": len(st.session_state.layers),
                "name": f"Layer {len(st.session_state.layers)}",
                "image": np.array(overlay_img),
                "visible": True,
                "opacity": 0.7
            })

        # Layer controls
        delete_idx = None
        for i, layer in enumerate(reversed(st.session_state.layers)):
            idx = len(st.session_state.layers) - 1 - i
            cols = st.columns([1,3,1,1,2,1])
            with cols[0]:
                st.session_state.layers[idx]["visible"] = st.checkbox(
                    " ", value=layer["visible"], key=f"visible_{idx}"
                )
                st.write("👁")
            with cols[1]:
                st.write(layer["name"])
            with cols[2]:
                if st.button("🔼", key=f"up_{idx}") and idx < len(st.session_state.layers)-1:
                    st.session_state.layers[idx], st.session_state.layers[idx+1] = (
                        st.session_state.layers[idx+1], st.session_state.layers[idx]
                    )
            with cols[3]:
                if st.button("🔽", key=f"down_{idx}") and idx > 0:
                    st.session_state.layers[idx], st.session_state.layers[idx-1] = (
                        st.session_state.layers[idx-1], st.session_state.layers[idx]
                    )
            with cols[4]:
                st.session_state.layers[idx]["opacity"] = st.slider(
                    "Opacity", 0.0, 1.0, layer["opacity"], step=0.05, key=f"opacity_{idx}"
                )
            with cols[5]:
                if layer["name"] != "Background":
                    if st.button("❌", key=f"delete_{idx}"):
                        delete_idx = idx

        # delete layer (ถ้ามี)
        if delete_idx is not None:
            st.session_state.layers.pop(delete_idx)

        # composite layers
        if st.session_state.layers:
            final = np.zeros_like(st.session_state.layers[0]["image"])
            for layer in st.session_state.layers:
                if layer["visible"]:
                    final = cv2.addWeighted(final, 1.0, layer["image"], layer["opacity"], 0)
            st.image(final, caption="ผลลัพธ์รวมเลเยอร์", channels="RGB", use_column_width=True)

            # download
            final_pil = Image.fromarray(final)
            buf = BytesIO()
            final_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="edited_image.png",
                mime="image/png"
            )
