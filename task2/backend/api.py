
"""
SOI ComfyUI API - Segmentation & Composition Migration
"""

import tempfile
import shutil
import random
from pathlib import Path
from typing import List, Optional
import time
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from segmentation_engine import SegmentationEngine
from data_manager import SessionManager, BACKGROUNDS
from email_service import EmailService
import comfyFunctions

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="SOI API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs for frontend access
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Initialize Engines
seg_engine = SegmentationEngine()
email_service = EmailService()

# --- Data Models ---

class ImageResource(BaseModel):
    id: str
    url: str

class SegmentationResponse(BaseModel):
    original: ImageResource
    stylized: List[ImageResource]

class CompositeRequest(BaseModel):
    silhouette_id: str
    background_id: str

class CompositeResponse(BaseModel):
    result: ImageResource

class EmailRequest(BaseModel):
    ids: List[str]
    email: str

class EmailResponse(BaseModel):
    status: str
    message: str

class ClearResponse(BaseModel):
    status: str

# --- Endpoints ---

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "test.html")

@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(image: UploadFile = File(...)):
    """
    API 1: Input image -> Returns 1 original silhouette + 2 stylized silhouettes (via ComfyUI).
    """
    try:
        # 1. Save Input
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            input_path = Path(tmp.name)

        input_id, saved_input_path = SessionManager.save_input_image(input_path)
        
        # 2. Generate Original Silhouette (Task 1 logic)
        logger.info("Extracting original silhouette...")
        sil_image = seg_engine.extract_silhouette(str(saved_input_path))
        
        # Save original silhouette
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as sil_tmp:
            sil_image.save(sil_tmp, format="PNG")
            sil_tmp_path = Path(sil_tmp.name)
            
        orig_sil_id, orig_sil_path = SessionManager.save_silhouette(sil_tmp_path, input_id, "orig")
        sil_tmp_path.unlink() # cleanup temp
        
        orig_resource = ImageResource(
            id=orig_sil_id,
            url=SessionManager.get_relative_url(orig_sil_path)
        )

        # 3. Generate Stylized Silhouettes (ComfyUI)
        stylized_resources = []
        try:
            logger.info("Submitting ComfyUI job...")
            job_id = comfyFunctions.submit_job(str(saved_input_path))
            
            # Wait for completion (blocking)
            logger.info(f"Waiting for ComfyUI job {job_id}...")
            result = comfyFunctions.wait_for_job(job_id, timeout=120.0)
            
            if result.status == comfyFunctions.JobStatus.FINISHED and result.generated_images:
                 # Move ComfyUI outputs to Session
                for i, rel_path in enumerate(result.generated_images):
                    filename = Path(rel_path).name
                    source_path = OUTPUTS_DIR / filename
                    
                    if source_path.exists():
                        # Requested Change: Segment the stylized output
                        logger.info(f"Segmenting stylized output {filename}...")
                        try:
                            stylized_seg_img = seg_engine.extract_silhouette(str(source_path))
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as style_tmp:
                                stylized_seg_img.save(style_tmp, format="PNG")
                                style_tmp_path = Path(style_tmp.name)
                            
                            style_id, saved_path = SessionManager.save_silhouette(style_tmp_path, input_id, f"style_{i}")
                            style_tmp_path.unlink()
                            
                            stylized_resources.append(ImageResource(
                                id=style_id,
                                url=SessionManager.get_relative_url(saved_path)
                            ))
                        except Exception as e:
                            logger.error(f"Failed to segment stylized output {filename}: {e}")
                            # Fallback: Save original unsegmented if segmentation fails? 
                            # Or skip? Let's skip to avoid confusion.
                            pass
            else:
                logger.warning(f"ComfyUI job failed or returned no images: {result.error}")

        except Exception as e:
            logger.error(f"ComfyUI generation failed: {e}")
            # Continue to return original even if Comfy fails

        # Cleanup input temp
        input_path.unlink()

        return SegmentationResponse(
            original=orig_resource,
            stylized=stylized_resources[:2] # Return what we have
        )

    except Exception as e:
        logger.exception("Error in /segment")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/composite", response_model=CompositeResponse)
async def compose_image(request: CompositeRequest):
    """
    API 2: Silhouette ID + Background ID -> Random composition.
    """
    try:
        # 1. Get Silhouette
        sil_path = SessionManager.get_silhouette_path(request.silhouette_id)
        if not sil_path:
            raise HTTPException(status_code=404, detail="Silhouette not found")
            
        # 2. Get Background
        bg_info = SessionManager.get_background_info(request.background_id)
        if not bg_info:
             raise HTTPException(status_code=404, detail="Background not found")
        
        bg_path = bg_info["path"]
        positions = bg_info["positions"]
        scale = bg_info.get("scale", 1.0) # Default to 1.0 if not present
        
        if not positions:
             raise HTTPException(status_code=500, detail="Background has no valid positions")

        # 3. Random Position
        pos = random.choice(positions)
        x, y = pos[0], pos[1]
        
        # 4. Composite
        # Using PIL
        from PIL import Image
        
        background = Image.open(bg_path).convert("RGBA")
        silhouette = Image.open(sil_path).convert("RGBA")
        
        # Apply scaling
        if scale != 1.0:
            new_width = int(silhouette.width * scale)
            new_height = int(silhouette.height * scale)
            silhouette = silhouette.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Paste centered at x,y
        paste_x = x - (silhouette.width // 2)
        paste_y = y - (silhouette.height // 2)
        
        # Create a new layer for silhouette matching background size
        sil_layer = Image.new("RGBA", background.size, (0,0,0,0))
        sil_layer.paste(silhouette, (paste_x, paste_y))
        
        final_comp = Image.alpha_composite(background, sil_layer)
        
        # 5. Save
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            final_comp.save(tmp, format="PNG")
            tmp_path = Path(tmp.name)
            
        comp_id, saved_path = SessionManager.save_composition(tmp_path)
        tmp_path.unlink()
        
        return CompositeResponse(result=ImageResource(
            id=comp_id,
            url=SessionManager.get_relative_url(saved_path)
        ))

    except Exception as e:
        logger.exception("Error in /composite")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/send-email", response_model=EmailResponse)
async def send_email(request: EmailRequest):
    """
    API 3: Send list of image IDs to email.
    """
    try:
        # Resolve IDs to paths
        paths = []
        for img_id in request.ids:
            # We need to find where the image is stored based on ID prefix or check all locations
            path = SessionManager.find_path_by_id(img_id)
            if path:
                paths.append(path)
            else:
                logger.warning(f"Image ID not found for email: {img_id}")

        if not paths:
            return EmailResponse(status="warning", message="No valid images found to send.")

        result = email_service.send_email(request.email, paths)
        return EmailResponse(status="success", message=result)
        
    except Exception as e:
        logger.exception("Error sending email")
        return EmailResponse(status="error", message=str(e))


@app.post("/clear", response_model=ClearResponse)
async def clear_data():
    """
    API 4: Clear all session data.
    """
    SessionManager.clear_session()
    return ClearResponse(status="success")


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces
    uvicorn.run(app, host="0.0.0.0", port=8000)
