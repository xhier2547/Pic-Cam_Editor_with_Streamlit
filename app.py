import streamlit as st
import cv2
import numpy as np
from PIL import Image
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
    uploaded_file = st.file_uploader("อัปโหลดไฟล์รูปภาพหลัก", type=["jpg", "png", "jpeg"], key="base")
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

        img = layer["image"]
        scale = layer.get("scale", 1.0)

        # ย่อ/ขยาย layer
        if scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        h, w = img.shape[:2]
        x, y = layer.get("pos_x", 0), layer.get("pos_y", 0)

        x_end, y_end = min(x + w, canvas_size[0]), min(y + h, canvas_size[1])
        lx_end, ly_end = x_end - x, y_end - y

        if x < canvas_size[0] and y < canvas_size[1]:
            roi = final[y:y_end, x:x_end]
            overlay = cv2.addWeighted(
                roi, 1 - layer["opacity"],
                img[:ly_end, :lx_end], layer["opacity"], 0
            )
            final[y:y_end, x:x_end] = overlay
    return final

# ---------------- PROCESSING ---------------- #
if image:
    processed = np.array(image)

    # Layout: Tools | Canvas | Layers
    col1, col2, col3 = st.columns([1, 2.5, 1.5])

    # -------- Tools -------- #
    with col1:
        st.markdown("### 🛠 Tools")
        apply_crop = st.checkbox("✂️ Crop")

    # -------- Canvas -------- #
    with col2:
        st.markdown("### 🖼 Canvas")

        if apply_crop:
            st.info("ลากกรอบ Crop แล้วกดปุ่มด้านล่างเพื่อยืนยัน")
            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color="#FF0000",
                aspect_ratio=None
            )

            if st.button("✅ Confirm Crop"):
                cropped_np = np.array(cropped_img)

                # ➡️ เพิ่มเป็นเลเยอร์ใหม่
                st.session_state.layers.append({
                    "id": len(st.session_state.layers),
                    "name": f"Layer Crop {len(st.session_state.layers)}",
                    "image": cropped_np,
                    "visible": True,
                    "opacity": 1.0,
                    "pos_x": 0,
                    "pos_y": 0,
                    "scale": 1.0
                })
                st.success("เพิ่มเลเยอร์ใหม่จาก Crop แล้ว ✅")

        # แสดงผลรวมเลเยอร์
        if st.session_state.layers:
            canvas_size = (st.session_state.layers[0]["image"].shape[1],
                           st.session_state.layers[0]["image"].shape[0])
            final = composite_layers(st.session_state.layers, canvas_size)
            st.image(final, caption="ผลลัพธ์รวมเลเยอร์", channels="RGB", use_column_width=True)
        else:
            st.image(processed, caption="ผลลัพธ์", channels="RGB", use_column_width=True)

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
                "pos_y": 0,
                "scale": 1.0
            })

        # add overlay
        overlay_file = st.file_uploader("➕ Add Overlay", type=["jpg", "png", "jpeg"], key=f"overlay_{len(st.session_state.layers)}")
        if overlay_file:
            overlay_img = Image.open(overlay_file).convert("RGB")
            st.session_state.layers.append({
                "id": len(st.session_state.layers),
                "name": f"Layer {len(st.session_state.layers)}",
                "image": np.array(overlay_img),
                "visible": True,
                "opacity": 0.7,
                "pos_x": 0,
                "pos_y": 0,
                "scale": 1.0
            })

        # Layer controls
        delete_idx = None
        for i, layer in enumerate(reversed(st.session_state.layers)):
            idx = len(st.session_state.layers) - 1 - i
            cols = st.columns([1, 2, 2, 2, 2, 2, 1])

            with cols[0]:
                st.session_state.layers[idx]["visible"] = st.checkbox(" ", value=layer["visible"], key=f"visible_{idx}")
            with cols[1]:
                st.image(layer["image"], width=50, channels="RGB")
            with cols[2]:
                st.write(layer["name"])
            with cols[3]:
                st.session_state.layers[idx]["pos_x"] = st.slider("X", -500, 500, layer.get("pos_x", 0), key=f"x_{idx}")
            with cols[4]:
                st.session_state.layers[idx]["pos_y"] = st.slider("Y", -500, 500, layer.get("pos_y", 0), key=f"y_{idx}")
            with cols[5]:
                st.session_state.layers[idx]["opacity"] = st.slider("Opacity", 0.0, 1.0, layer["opacity"], step=0.05, key=f"opacity_{idx}")
                st.session_state.layers[idx]["scale"] = st.slider("Scale", 0.1, 3.0, layer.get("scale", 1.0), step=0.1, key=f"scale_{idx}")
            with cols[6]:
                if layer["name"] != "Background":
                    if st.button("❌", key=f"delete_{idx}"):
                        delete_idx = idx
                # Save button
                if st.button("💾", key=f"save_{idx}"):
                    pil_img = Image.fromarray(layer["image"])
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    st.download_button(
                        label=f"Download {layer['name']}",
                        data=buf.getvalue(),
                        file_name=f"{layer['name']}.png",
                        mime="image/png",
                        key=f"dl_{idx}"
                    )

        if delete_idx is not None:
            st.session_state.layers.pop(delete_idx)

        # Download all layers merged
        if st.session_state.layers:
            canvas_size = (st.session_state.layers[0]["image"].shape[1],
                           st.session_state.layers[0]["image"].shape[0])
            final = composite_layers(st.session_state.layers, canvas_size)
            final_pil = Image.fromarray(final)
            buf = BytesIO()
            final_pil.save(buf, format="PNG")
            st.download_button(
                label="Download All Layers (Merged)",
                data=buf.getvalue(),
                file_name="edited_image.png",
                mime="image/png"
            )
