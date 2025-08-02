import streamlit as st
import requests
from PIL import Image
import io

FASTAPI_BASE_URL = "http://fastapi-app:8000"  # Update if hosted elsewhere

st.set_page_config(page_title="Inference Dashboard", layout="wide")
st.title("🚦 Object Detection & Tracking")

# Sidebar navigation
option = st.sidebar.radio("Choose Task", ["Image Detection", "Video Tracking"])

if option == "Image Detection":
    st.header("📸 Image Detection")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    if uploaded_image and st.button("Run Detection"):
        with st.spinner("Running object detection..."):
            files = {
                "file": (uploaded_image.name, uploaded_image, uploaded_image.type)
            }
            data = {"confidence": confidence}
            response = requests.post(f"{FASTAPI_BASE_URL}/detect", files=files, data=data)

        if response.status_code == 200:
            st.success("✅ Detection complete!")
            #st.write(f"Response content type: {response.headers.get('Content-Type')}")
            #st.write(f"Response content size: {len(response.content)} bytes")

            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption="Detected Output", use_column_width=True)

            inf_time = response.headers.get("X-Inference-Time-in-seconds")
            if inf_time:
                st.metric("Inference Time (sec)", inf_time)
        else:
            st.error(f"❌ Detection failed: {response.text}")

elif option == "Video Tracking":
    st.header("🎞️ Video Tracking")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    if uploaded_video and st.button("Run Tracking"):
        with st.spinner("Running object tracking..."):
            files = {
                "file": (uploaded_video.name, uploaded_video, uploaded_video.type or "video/mp4")
            }
            data = {"confidence": confidence}
            response = requests.post(f"{FASTAPI_BASE_URL}/track_video", files=files, data=data)

        if response.status_code == 200:
            st.success("✅ Tracking complete!")

            inf_time = response.headers.get("X-Inference-Time-in-seconds")
            if inf_time:
                st.metric("Inference Time (sec)", float(inf_time))

            st.download_button(
                label="📥 Download Tracked Video",
                data=response.content,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )
        else:
            st.error(f"❌ Tracking failed: {response.text}")
