import streamlit as st
import requests
import os
import time
import numpy as np
from PIL import Image

# Backend URL (can be environment variable)
API_URL = "https://unified-autoatlas.onrender.com"

st.set_page_config(page_title="AutoAtlas Dashboard", layout="wide")

st.title("🧠 AutoAtlas Medical Image Analysis")
st.markdown("### Integrated Tech Stack: PyTorch, AUTOATLAS, FastAPI")

# Session Management
if "session_id" not in st.session_state:
    try:
        resp = requests.post(f"{API_URL}/session").json()
        st.session_state.session_id = resp["session_id"]
    except:
        st.error("Failed to connect to Backend API. Please ensure FastAPI is running.")
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
                
                # Mock multi-part form data upload via requests
                # In real scenario, we'd pass the actual file object
                files_payload = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_URL}/process", data=data, files=files_payload)
                
                if resp.status_code == 200:
                    st.success("Transformation Complete!")
                    res = resp.json()
                    st.session_state.last_result = res
                else:
                    st.error("Processing Failed.")

    with col_r:
        if "last_result" in st.session_state:
            res = st.session_state.last_result
            st.subheader("Processing Results")
            
            # Display Original (from local data if available, or fetch)
            # Display Synthetic and Segmented
            if res.get("translated_path"):
                st.markdown(f"**Synthetic { 'CT' if modality == 'MRI' else 'MRI' }**")
                # Fetching via file endpoint
                t_img_resp = requests.get(f"{API_URL}/files/{res['translated_path']}", stream=True)
                if t_img_resp.status_code == 200:
                    st.image(Image.open(t_img_resp.raw), use_column_width=True)
            
            if res.get("segmented_path"):
                st.markdown("**Segmentation Map (AutoAtlas)**")
                s_img_resp = requests.get(f"{API_URL}/files/{res['segmented_path']}", stream=True)
                if s_img_resp.status_code == 200:
                    st.image(Image.open(s_img_resp.raw), use_column_width=True)

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
            # Generating a simple 3D mesh-like scatter for demonstration
            t = np.linspace(0, 2*np.pi, 100)
            x = np.sin(t)
            y = np.cos(t)
            z = t / (2*np.pi)
            ax.plot(x, y, z, label='Reconstruction Path', color='cyan')
            ax.set_title("3D Segmentation Reconstruction (Matplotlib)")
            st.pyplot(fig)

with tab_history:
    st.subheader("Session History")
    hist_resp = requests.get(f"{API_URL}/history/{st.session_state.session_id}")
    if hist_resp.status_code == 200:
        history = hist_resp.json()["history"]
        for item in history:
            with st.expander(f"📄 {item['filename']} | {item['modality']} | {item['uploaded_at']}"):
                hc1, hc2 = st.columns(2)
                with hc1:
                    if item.get("translated_path"):
                        st.markdown("Synthetic Output:")
                        st.image(f"{API_URL}/files/{item['translated_path']}")
                with hc2:
                    if item.get("segmented_path"):
                        st.markdown("Autoatlas Segmentation:")
                        st.image(f"{API_URL}/files/{item['segmented_path']}")
    else:
        st.write("No history available.")
