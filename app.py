import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pandas as pd

# =========================
# Page Config (FIRST)
# =========================
st.set_page_config(
    page_title="Electronics Component Detection",
    layout="wide"
)

# =========================
# Device Selection
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"💻 Using device: {device.upper()}")

# =========================
# Load YOLO Model
# =========================
@st.cache_resource
def load_model():
    model_path = (
        "C:/Users/Bhaktesh/electronic_component_detection/"
        "runs/detect/train7/weights/best.pt"
    )
    model = YOLO(model_path)
    model.to(device)
    return model

model = load_model()

# =========================
# Title
# =========================
st.title("🔌 Electronics Component Detection System")
st.markdown(
    "Upload an image, capture from camera, or use live detection "
    "to identify electronic components."
)

# =========================
# Detection Function
# =========================
def detect(image):
    results = model(image)
    annotated = results[0].plot()
    return annotated, results

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs([
    "📁 Upload Image",
    "📸 Capture Image (Camera)",
    "🎥 Live Camera Detection"
])

# ============================================================
# TAB 1: IMAGE UPLOAD
# ============================================================
with tab1:
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        file_bytes = np.asarray(
            bytearray(uploaded.read()),
            dtype=np.uint8
        )
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(
            image_rgb,
            caption="Uploaded Image",
            use_column_width=True
        )

        if st.button("🔍 Detect Components"):
            annotated, results = detect(image)

            st.image(
                annotated[:, :, ::-1],
                caption="Detection Result",
                use_column_width=True
            )

            boxes = results[0].boxes
            if boxes is not None:
                df = boxes.data.cpu().numpy()
                st.dataframe(
                    pd.DataFrame(
                        df,
                        columns=[
                            "x1", "y1", "x2", "y2",
                            "confidence", "class"
                        ]
                    )
                )

# ============================================================
# TAB 2: CAPTURE SINGLE IMAGE (OpenCV ONLY)
# ============================================================
with tab2:
    st.subheader("📸 Capture Image from Camera")

    if st.button("Open Camera"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Camera not accessible")
        else:
            ret, frame = cap.read()
            cap.release()

            if ret:
                st.image(
                    frame,
                    channels="BGR",
                    caption="Captured Image",
                    use_column_width=True
                )

                if st.button("🔍 Detect on Captured Image"):
                    annotated, results = detect(frame)

                    st.image(
                        annotated,
                        channels="BGR",
                        caption="Detection Result",
                        use_column_width=True
                    )
            else:
                st.error("❌ Failed to capture image")

# ============================================================
# TAB 3: LIVE CAMERA (OpenCV STREAM)
# ============================================================
with tab3:
    st.subheader("🎥 Live Camera Detection")

    if "live" not in st.session_state:
        st.session_state.live = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Live Camera"):
            st.session_state.live = True

    with col2:
        if st.button("⏹ Stop Live Camera"):
            st.session_state.live = False

    frame_window = st.image([])

    if st.session_state.live:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Failed to access camera")
        else:
            while st.session_state.live:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Camera frame read failed")
                    break

                results = model(frame)
                annotated = results[0].plot()

                frame_window.image(
                    annotated,
                    channels="BGR"
                )

            cap.release()
