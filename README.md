# 🖼 Explore the rapid prototype in AI Project, Image Editor with Layers + Webcam  By 6610110688

โปรเจคนี้เป็น **Web App ที่ใช้ Streamlit** สำหรับแก้ไขรูปภาพแบบง่าย ๆ คล้าย Photoshop / Canva  
รองรับการทำงานกับ **Layers** (Background + Overlay + Crop) และยังสามารถใช้ **Webcam Realtime** พร้อมการตรวจจับวัตถุด้วย **YOLOv5** ได้ด้วย

---

## ✨ Features

### 📂 Input
- เลือกภาพจาก:
  - Upload Image (อัปโหลดไฟล์จากเครื่อง)
  - Image URL (โหลดภาพจากลิงก์อินเตอร์เน็ต)
  - Webcam Realtime (เปิดกล้อง Notebook)

### 🛠 Tools
- ✂️ **Crop** → เลือกกรอบด้วยเมาส์แล้วกด Confirm เพื่อสร้างเลเยอร์ใหม่
- 🔎 **YOLOv5 Detection** → ตรวจจับใบหน้า/วัตถุผ่านโมเดล YOLOv5s
- 🔆 **Brightness / Contrast Control** (เฉพาะ Webcam Mode)
- 💨 **Blur (Gaussian Blur)** (เฉพาะ Webcam Mode)
- 📏 **Move / Scale / Opacity** → ควบคุมแต่ละเลเยอร์

### 🗂 Layers
- สามารถมีหลายเลเยอร์ (Background, Overlay, Crop Layers)
- Toggle 👁 เพื่อเปิด/ปิดการมองเห็น
- ปรับ Opacity (0–1.0)
- ปรับ Scale (0.1–3.0)
- ปรับตำแหน่ง X,Y
- ลบเลเยอร์ได้ ❌
- ดาวน์โหลดแต่ละเลเยอร์ หรือภาพรวมทั้งหมด

### 📊 Graphs
- แสดง **Histogram ของภาพสุดท้าย (Final Image)**  
  เพื่อดูการกระจายของ Pixel Intensity (RGB หรือ Grayscale)

---

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-repo/photoshop-lite-streamlit.git
cd photoshop-lite-streamlit
```

### 2. สร้าง Virtual Environment และติดตั้ง Dependencies
```bash
python -m venv .venv
source .venv/bin/activate     # (Linux/Mac)
.venv\Scripts\activate        # (Windows)

pip install -r requirements.txt
```

### 3. Run App
```bash
streamlit run app.py
```




---
### Project by 6610110688 <มารสวรรค์>

