import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from backend.database import init_db, create_session, save_upload, save_result, get_session_history
from backend.models_dl import run_monai_segmentation, run_cyclegan_translation

app = FastAPI(title="AutoAtlas AI API")

# Initialize DB
init_db()

UPLOAD_DIR = "data/uploads"
RESULT_DIR = "data/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.post("/session")
def new_session():
    session_id = create_session()
    return {"session_id": session_id}

@app.post("/process")
async def process_image(
    session_id: str = Form(...),
    modality: str = Form(...),
    atlas_choice: str = Form(...),
    enable_translation: bool = Form(True),
    file: UploadFile = File(...)
):
    # 1. Save File
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Save Upload Record
    upload_id = save_upload(session_id, file.filename, modality, file_path)
    
    # Session-specific result directory
    session_res_dir = os.path.join(RESULT_DIR, session_id)
    os.makedirs(session_res_dir, exist_ok=True)
    
    # 3. Translation (CycleGAN)
    translated_path = None
    if enable_translation:
        target = "CT" if modality == "MRI" else "MRI"
        translated_path = os.path.join(session_res_dir, f"trans_{target}_{file.filename}.png")
        run_cyclegan_translation(file_path, translated_path, mode="mri2ct" if modality == "MRI" else "ct2mri")
    
    # 4. Segmentation (MONAI)
    input_for_seg = translated_path if enable_translation else file_path
    segmented_path = os.path.join(session_res_dir, f"seg_{atlas_choice}_{file.filename}.png")
    run_monai_segmentation(input_for_seg, segmented_path)
    
    # 5. Save Results
    save_result(upload_id, translated_path, segmented_path, atlas_choice)
    
    # 6. Generate Diagnostic Report (Groq)
    from backend.models_dl import generate_diagnostic_report
    report = generate_diagnostic_report(modality, atlas_choice)
    
    return {
        "upload_id": upload_id,
        "translated_path": translated_path,
        "segmented_path": segmented_path,
        "session_id": session_id,
        "report": report
    }

@app.get("/history/{session_id}")
def get_history(session_id: str):
    history = get_session_history(session_id)
    return {"history": history}

@app.get("/files/{path:path}")
def get_file(path: str):
    # Standard security check for path traversal should be here
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
