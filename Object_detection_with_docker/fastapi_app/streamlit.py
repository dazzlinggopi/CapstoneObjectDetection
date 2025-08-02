import streamlit as st
import requests
from PIL import Image
import io

FASTAPI_BASE_URL = "http://localhost:8000"  # Change if hosted remotely

st.set_page_config(page_title="Inference Dashboard", layout="wide")
st.title("🚦 Object Detection & Tracking")

# 📌 Sidebar for Navigation
option = st.sidebar.radio("Choose Task", ["Image Detection", "Video Tracking"])

if option == "Image Detection":
    st.header("📸 Image Detection")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.15)

    if uploaded_image and st.button("Run Detection"):
        with st.spinner("Running object detection in images..."):
            response = requests.post(
                f"{FASTAPI_BASE_URL}/detect",
                files={"file": uploaded_image.getvalue()},
                data={"confidence": confidence}
            )

        if response.status_code == 200:
            st.success("Detection complete!")
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption="Detected Image", use_column_width=True)

            # 🕒 Show inference time from header
            inf_time = response.headers.get("X-Inference-Time in seconds")
            if inf_time:
                st.metric("Inference Time (sec)", inf_time)
        else:
            st.error(f"Detection failed: {response.text}")

elif option == "Video Tracking":
    st.header("🎞️ Video Tracking")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.15)

    if uploaded_video and st.button("Run Tracking"):
        with st.spinner("Running object tracking in video..."):
            # Construct files dict with full 3-tuple: (filename, file object, MIME type)
            files = {
                "file": (uploaded_video.name, uploaded_video, "video/mp4")  # Adjust MIME if needed
            }
            data = {"confidence": confidence}

            response = requests.post(
                f"{FASTAPI_BASE_URL}/track_video",
                files=files,
                data=data
            )

        if response.status_code == 200:
            st.success("Tracking complete!")

            print(response.headers)
            inf_time = response.headers.get("X-Inference-Time in seconds")
            if inf_time:
                st.metric("Inference Time (sec)", float(inf_time))

            st.download_button(
                label="📥 Download Tracked Video",
                data=response.content,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )
        else:
            st.error(f"Tracking failed: {response.text}")
