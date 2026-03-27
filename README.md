# 🧠 Cross-Domain MRI ↔ CT Mapping App

An end-to-end medical image processing web application built with **Streamlit**, **FastAPI**, and **SQLite**. It accepts either an MRI or CT scan as input, performs cross-domain image translation (MRI → CT or CT → MRI), and returns segmented output images using **AutoAtlas** (MONAI) as the core image processing and atlas-based segmentation framework.

---

## 🏗️ Technology Stack

| Domain | Libraries / Tools |
| --- | --- |
| **Deep Learning** | PyTorch 2.0+, MONAI 1.x, CycleGAN |
| **Medical Imaging** | NiBabel (NIfTI), SimpleITK (DICOM), pydicom |
| **Backend / API** | FastAPI (REST) on port 8000, Uvicorn |
| **Frontend / UI** | Streamlit on port 8501, Requests, PIL |
| **Data Science** | NumPy, Pandas, scikit-learn |
| **Visualization** | Matplotlib 3D, Plotly placeholders |
| **Database** | SQLite (session history) via `backend/database.py` |
| **Deployment** | Docker, Python .venv |

---

## ⚙️ Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages include: `streamlit`, `torch`, `monai`, `fastapi`, `uvicorn`, `nibabel`, `SimpleITK`, `opencv-python`, `numpy`, `groq`, `python-dotenv`

---

## 🚀 Setup and Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-folder>

# 2. Setup Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Backend (FastAPI)
python -m backend.main

# 5. Launch the Frontend (Streamlit)
streamlit run frontend/app.py
```

Then open [localhost:8501](http://localhost:8501) in your browser.

---

## ✨ Features

### 🖥️ Frontend / UI

- Streamlit-based dashboard with dual tabs for Processing and History.
- Drag-and-drop file support for NIfTI and common image formats.
- Real-time AI Diagnostic reports integrated with **Groq API**.

### 🗄️ Backend / Database

- **FastAPI** robust service layer handling file uploads and model execution.
- **SQLite** backend with robust path handling and context-managed connections.
- Persistent session history tracking every generation event.

### 🔬 Medical Imaging Pipeline

- Modality Translation (CycleGAN) for MRI ↔ CT mapping.
- Automated Segmentation (MONAI) with atlas selection (MNI152, BodyAtlas).
- Support for 3D Volume reconstructions and visualization.

### 🐳 Deployment

- Fully containerized with **Docker** support.
- Environment-controlled via `.env` (Groq API keys).

---

## 📁 Project Structure

```text
.
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── database.py         # SQLite connection & schema
│   ├── models_dl.py        # CycleGAN & MONAI logic
│   └── autoatlas.db        # SQLite database file
├── frontend/
│   └── app.py              # Streamlit dashboard
├── data/
│   ├── uploads/            # Temporary storage for uploaded scans
│   └── results/            # Processed outputs (synthetic & segmented)
├── requirements.txt        # Project dependencies
├── Dockerfile              # Containerization instructions
└── .env                    # Environment variables (API Keys)
```

---

## 📌 Notes

- Ensure GPU support is available for optimal inference speed (`torch.cuda.is_available()`).
- All session logs and generated images are persisted and can be viewed in the "History" tab of the dashboard.
- Database connections are optimized for Windows path environments and include foreign key enforcement.
