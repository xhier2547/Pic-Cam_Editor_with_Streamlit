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

# ---------------- ฟังก์ชันรวมเลเยอร์ ---------------- #
def composite_layers(layers, canvas_size):
    final = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    for layer in layers:
        if not layer["visible"]:
            continue

        h, w = layer["image"].shape[:2]
        x, y = layer["pos_x"], layer["pos_y"]

        # กันไม่ให้ออกนอกขอบ
        x_end, y_end = min(x+w, canvas_size[0]), min(y+h, canvas_size[1])
        lx_end, ly_end = x_end-x, y_end-y

        if x < canvas_size[0] and y < canvas_size[1]:
            roi = final[y:y_end, x:x_end]
            overlay = cv2.addWeighted(
                roi, 1-layer["opacity"],
                layer["image"][:ly_end, :lx_end], layer["opacity"], 0
            )
            final[y:y_end, x:x_end] = overlay
    return final

# ---------------- PROCESSING ---------------- #
if image:
    processed = np.array(image)

    # Layout: Tools | Canvas | Layers
    col1, col2, col3 = st.columns([1,2.5,1.5])

    # -------- Tools -------- #
    with col1:
        st.markdown("### 🛠 Tools")
        apply_crop   = st.checkbox("✂️ Crop")
        apply_gray   = st.checkbox("⚫ Grayscale")
        apply_canny  = st.checkbox("📏 Canny")
        apply_rotate = st.checkbox("🔄 Rotate")
        apply_detect = st.checkbox("🔎 Detect Face")

    # -------- Canvas -------- #
    with col2:
        st.markdown("### 🖼 Canvas")

        img_cv = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        # Crop tool
        if apply_crop:
            st.info("ลากกรอบ Crop แล้วกดปุ่มด้านล่างเพื่อยืนยันหรือยกเลิก")
            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color="#FF0000",
                aspect_ratio=None
            )

            col_crop = st.columns([1,1])
            with col_crop[0]:
                if st.button("✅ Confirm Crop"):
                    cropped_np = np.array(cropped_img)
                    st.session_state.layers.append({
                        "id": len(st.session_state.layers),
                        "name": f"Layer Crop {len(st.session_state.layers)}",
                        "image": cropped_np,
                        "visible": True,
                        "opacity": 1.0,
                        "pos_x": 0,
                        "pos_y": 0
                    })
                    st.success("เพิ่มเลเยอร์ใหม่จาก Crop แล้ว ✅")

        # -------- Filters (apply to preview only) -------- #
        preview_img = processed.copy()

        if apply_gray:
            preview_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        if apply_canny:
            t1 = st.slider("Threshold1", 0, 255, 100)
            t2 = st.slider("Threshold2", 0, 255, 200)
            preview_img = cv2.Canny(img_cv, t1, t2)

        if apply_rotate:
            angle = st.slider("หมุน (degree)", -180, 180, 0)
            h, w = preview_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            preview_img = cv2.warpAffine(img_cv, M, (w, h))

        if apply_detect:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            preview_img = img_cv

        # -------- Show Result -------- #
        if st.session_state.layers:
            canvas_size = (image.width, image.height)
            final = composite_layers(st.session_state.layers, canvas_size)
            st.image(final, caption="ผลลัพธ์รวมเลเยอร์", channels="RGB", use_column_width=True)
        else:
            st.image(preview_img, caption="ผลลัพธ์ (Preview)", channels="RGB" if preview_img.ndim==3 else "GRAY", use_column_width=True)

    # -------- Layers -------- #
    with col3:
        st.markdown("### 🗂 Layers Panel")

        # init background layer
        if not any(l["name"] == "Background" for l in st.session_state.layers):
            st.session_state.layers.append({
                "id": len(st.session_state.layers),
                "name": "Background",
                "image": np.array(image),
                "visible": True,
                "opacity": 1.0,
                "pos_x": 0,
                "pos_y": 0
            })

        # add overlay
        overlay_file = st.file_uploader("เพิ่มเลเยอร์ใหม่ (Overlay)", type=["jpg","png","jpeg"], key=f"overlay_{len(st.session_state.layers)}")
        if overlay_file:
            overlay_img = Image.open(overlay_file).convert("RGB")
            overlay_np = np.array(overlay_img)
            st.session_state.layers.append({
                "id": len(st.session_state.layers),
                "name": f"Layer {len(st.session_state.layers)}",
                "image": overlay_np,
                "visible": True,
                "opacity": 0.7,
                "pos_x": 0,
                "pos_y": 0
            })

        # Layer controls
        delete_idx = None
        for i, layer in enumerate(reversed(st.session_state.layers)):
            idx = len(st.session_state.layers) - 1 - i
            cols = st.columns([1,2,2,1,1,2,1])  
            with cols[0]:
                st.session_state.layers[idx]["visible"] = st.checkbox(
                    " ", value=layer["visible"], key=f"visible_{idx}"
                )
                st.write("👁")
            with cols[1]:
                st.image(layer["image"], width=50, channels="RGB")
            with cols[2]:
                st.write(layer["name"])
            with cols[5]:
                st.session_state.layers[idx]["opacity"] = st.slider(
                    "Opacity", 0.0, 1.0, layer["opacity"], step=0.05, key=f"opacity_{idx}"
                )
            with cols[6]:
                if layer["name"] != "Background":
                    if st.button("❌", key=f"delete_{idx}"):
                        delete_idx = idx

        if delete_idx is not None:
            st.session_state.layers.pop(delete_idx)

        # download
        if st.session_state.layers:
            canvas_size = (image.width, image.height)
            final = composite_layers(st.session_state.layers, canvas_size)
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
