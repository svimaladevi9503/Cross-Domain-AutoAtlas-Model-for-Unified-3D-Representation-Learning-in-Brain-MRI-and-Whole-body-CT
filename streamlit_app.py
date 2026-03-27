import streamlit as st
import requests
import os
import time
import numpy as np
from PIL import Image

# Backend URL - for Streamlit Cloud, set this as an environment variable
API_URL = os.getenv("API_URL", "http://localhost:8001")

st.set_page_config(page_title="AutoAtlas Dashboard", layout="wide")

st.title("🧠 AutoAtlas Medical Image Analysis")
st.markdown("### Integrated Tech Stack: PyTorch, AUTOATLAS, FastAPI")

# Session Management
if "session_id" not in st.session_state:
    try:
        resp = requests.post(f"{API_URL}/session").json()
        st.session_state.session_id = resp["session_id"]
    except Exception as e:
        st.error(f"Failed to connect to Backend API at {API_URL}. Please ensure FastAPI is running and API_URL is set correctly.")
        st.error(f"Error: {str(e)}")
        st.stop()

# Dashboard Tabs
tab_app, tab_history = st.tabs(["🚀 Process Scan", "🕒 History"])

with tab_app:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Upload & Config")
        uploaded_file = st.file_uploader("Upload NIfTI/DICOM or Image File", type=['nii', 'nii.gz', 'dcm', 'png', 'jpg'])
        modality = st.selectbox("Modality", ["MRI", "CT"])
        atlas = st.selectbox("Atlas Template", ["MNI152", "BodyAtlas"])
        enable_trans = st.checkbox("Enable Cross-Domain Translation", value=True)

        if uploaded_file and st.button("Run Transformation Pipeline"):
            with st.spinner("Processing via FastAPI..."):
                files = {"file": uploaded_file.getvalue()}
                data = {
                    "session_id": st.session_state.session_id,
                    "modality": modality,
                    "atlas_choice": atlas,
                    "enable_translation": str(enable_trans).lower()
                }

                # Multi-part form data upload via requests
                files_payload = {"file": (uploaded_file.name, uploaded_file.getvalue())}

                try:
                    resp = requests.post(f"{API_URL}/process", data=data, files=files_payload, timeout=300)

                    if resp.status_code == 200:
                        st.success("Transformation Complete!")
                        res = resp.json()
                        st.session_state.last_result = res
                    else:
                        st.error(f"Processing Failed. Status: {resp.status_code}")
                        st.error(f"Response: {resp.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {str(e)}")

    with col_r:
        if "last_result" in st.session_state:
            res = st.session_state.last_result
            st.subheader("Processing Results")

            # Display Synthetic and Segmented
            if res.get("translated_path"):
                st.markdown(f"**Synthetic { 'CT' if modality == 'MRI' else 'MRI' }**")
                try:
                    t_img_resp = requests.get(f"{API_URL}/files/{res['translated_path']}", stream=True, timeout=30)
                    if t_img_resp.status_code == 200:
                        st.image(Image.open(t_img_resp.raw), use_column_width=True)
                    else:
                        st.error("Failed to load synthetic image")
                except Exception as e:
                    st.error(f"Error loading synthetic image: {str(e)}")

            if res.get("segmented_path"):
                st.markdown("**Segmentation Map (AutoAtlas)**")
                try:
                    s_img_resp = requests.get(f"{API_URL}/files/{res['segmented_path']}", stream=True, timeout=30)
                    if s_img_resp.status_code == 200:
                        st.image(Image.open(s_img_resp.raw), use_column_width=True)
                    else:
                        st.error("Failed to load segmentation image")
                except Exception as e:
                    st.error(f"Error loading segmentation image: {str(e)}")

            # Display Diagnostic Report
            if res.get("report"):
                st.markdown("---")
                st.subheader("📝 AI Diagnostic Report (Groq)")
                st.info(res["report"])

            # 3D Visualization Placeholder
            st.markdown("---")
            st.subheader("📊 3D Volume Visualization")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            # Mock 3D data
            t = np.linspace(0, 2*np.pi, 100)
            x = np.sin(t)
            y = np.cos(t)
            z = t / (2*np.pi)
            ax.plot(x, y, z, label='Reconstruction Path', color='cyan')
            ax.set_title("3D Segmentation Reconstruction (Matplotlib)")
            st.pyplot(fig)

with tab_history:
    st.subheader("Session History")
    try:
        hist_resp = requests.get(f"{API_URL}/history/{st.session_state.session_id}", timeout=30)
        if hist_resp.status_code == 200:
            history = hist_resp.json()["history"]
            for item in history:
                with st.expander(f"📄 {item['filename']} | {item['modality']} | {item['uploaded_at']}"):
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        if item.get("translated_path"):
                            st.markdown("Synthetic Output:")
                            try:
                                st.image(f"{API_URL}/files/{item['translated_path']}")
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                    with hc2:
                        if item.get("segmented_path"):
                            st.markdown("Autoatlas Segmentation:")
                            try:
                                st.image(f"{API_URL}/files/{item['segmented_path']}")
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
        else:
            st.write("No history available.")
    except Exception as e:
        st.error(f"Failed to load history: {str(e)}")