"""
SOI OpenPose API - Segmentation & Composition
No ComfyUI. Pose detection via OpenPose (openpose_engine.py) instead of MediaPipe.
"""

import json
import os
import glob
import tempfile
import shutil
import random
from pathlib import Path
from typing import List, Optional
import logging
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from segmentation_engine import SegmentationEngine
from data_manager import SessionManager, BACKGROUNDS, BACKGROUNDS_DIR
from email_service import EmailService
from classical_warping import ClassicalWarpingEngine, get_default_engine
from openpose_engine import OpenPoseEngine

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API2")

app = FastAPI(title="SOI API v2 (OpenPose)", version="2.0.0")

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

# Mount backgrounds
BACKGROUNDS_DIR.mkdir(exist_ok=True)
app.mount("/backgrounds", StaticFiles(directory=str(BACKGROUNDS_DIR)), name="backgrounds")

# Initialize Engines
seg_engine = SegmentationEngine()
email_service = EmailService()

# Lazy-initialized engines
_openpose_engine: Optional[OpenPoseEngine] = None
_classical_warper: Optional[ClassicalWarpingEngine] = None

# OpenPose root directory.
# Override via the OPENPOSE_ROOT environment variable, or set a sensible default below.
# The directory must contain bin/OpenPoseDemo.exe and the model/ folder.
OPENPOSE_ROOT = Path(os.environ.get("OPENPOSE_ROOT", r"C:\openpose"))


def get_engines():
    """Lazy-initialize OpenPose and classical warping engines."""
    global _openpose_engine, _classical_warper
    if _openpose_engine is None:
        logger.info(f"Initializing OpenPose engine from: {OPENPOSE_ROOT}")
        _openpose_engine = OpenPoseEngine(openpose_root_dir=OPENPOSE_ROOT)
    if _classical_warper is None:
        _classical_warper = get_default_engine()
    return _openpose_engine, _classical_warper


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helper: Run OpenPose and return result dict compatible with classical_warping
# ---------------------------------------------------------------------------

def run_openpose_on_image(image_path: Path) -> dict:
    """
    Runs OpenPose on a single image and returns a pose-result dict in the same
    format as PoseEstimator.detect_poses():

        {
            "people": [ { "pose_keypoints_2d": [...], "hand_left_keypoints_2d": [...], ... }, ... ],
            "canvas_width": int,
            "canvas_height": int,
        }

    OpenPose already outputs OpenPose 25-point format natively, so no keypoint
    conversion is needed unlike the MediaPipe path.
    """
    op_engine, _ = get_engines()

    with tempfile.TemporaryDirectory() as input_tmpdir, \
         tempfile.TemporaryDirectory() as output_tmpdir:

        input_tmp = Path(input_tmpdir)
        output_tmp = Path(output_tmpdir)

        # Copy image into a clean temp folder so OpenPose can process it
        dest_image = input_tmp / image_path.name
        shutil.copy(str(image_path), str(dest_image))

        logger.info(f"Running OpenPose on {dest_image.name} ...")
        op_engine.extract_pose(input_tmp, output_tmp)

        # OpenPose writes <image_stem>_keypoints.json into output_tmp
        json_files = list(output_tmp.glob("*_keypoints.json"))
        if not json_files:
            logger.warning("OpenPose produced no keypoint JSON files.")
            img = Image.open(str(image_path))
            return _empty_pose_result(*img.size)

        with open(json_files[0], "r") as f:
            op_data = json.load(f)

    # Get image dimensions for canvas info
    img = Image.open(str(image_path))
    width, height = img.size

    people = op_data.get("people", [])
    # Strip person_id wrapper so the list entries look like PoseEstimator output
    cleaned_people = []
    for p in people:
        cleaned_people.append({
            "pose_keypoints_2d":       p.get("pose_keypoints_2d", []),
            "hand_left_keypoints_2d":  p.get("hand_left_keypoints_2d", []),
            "hand_right_keypoints_2d": p.get("hand_right_keypoints_2d", []),
            "face_keypoints_2d":       p.get("face_keypoints_2d", []),
        })

    return {
        "people": cleaned_people,
        "canvas_width": width,
        "canvas_height": height,
    }


def _empty_pose_result(width: int, height: int) -> dict:
    return {"people": [], "canvas_width": width, "canvas_height": height}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "test.html")


@app.get("/backgrounds")
async def get_backgrounds():
    """Returns a list of available backgrounds."""
    bg_list = []
    for bg_id, bg_info in BACKGROUNDS.items():
        bg_list.append({
            "id": bg_info["id"],
            "title": bg_info["filename"].split(".")[0].capitalize(),
            "url": f"/backgrounds/{bg_info['filename']}"
        })
    return bg_list


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    image: UploadFile = File(...),
    use_classical_warping: bool = Form(True),   # kept for test.html compatibility; always True
    parts_to_warp: Optional[str] = Form(None),
):
    """
    API 1: Input image → original silhouette + OpenPose-warped silhouette(s).

    Args:
        image:                Uploaded image file.
        use_classical_warping: Accepted for API compatibility with test.html;
                               always uses OpenPose + classical warping in api2.
        parts_to_warp:        Comma-separated body parts to warp.
                              Groups: arms, legs, feet, hands
                              Individual: head, neck, torso, right_upper_arm, etc.
                              Leave empty to warp all parts.
    """
    try:
        # 1. Save uploaded file to a temp location
        suffix = Path(image.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            input_path = Path(tmp.name)

        input_id, saved_input_path = SessionManager.save_input_image(input_path)

        # 2. Extract original silhouette (background removal)
        logger.info("Extracting original silhouette...")
        sil_image = seg_engine.extract_silhouette(str(saved_input_path))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as sil_tmp:
            sil_image.save(sil_tmp, format="PNG")
            sil_tmp_path = Path(sil_tmp.name)

        orig_sil_id, orig_sil_path = SessionManager.save_silhouette(sil_tmp_path, input_id, "orig")
        sil_tmp_path.unlink()

        orig_resource = ImageResource(
            id=orig_sil_id,
            url=SessionManager.get_relative_url(orig_sil_path)
        )

        # 3. Run OpenPose → get pose keypoints (only when warping is requested)
        stylized_resources = []
        if not use_classical_warping:
            logger.info("Warping not requested, skipping OpenPose.")
        else:
            try:
                logger.info("Running OpenPose for pose detection...")
                pose_result = run_openpose_on_image(saved_input_path)

                _, cw = get_engines()

                # Parse parts_to_warp
                parts_list = None
                if parts_to_warp:
                    parts_list = [p.strip() for p in parts_to_warp.split(",")]
                    # If the selection covers every template part, treat as full warp
                    # (transparent canvas, no partial-overlay on original silhouette)
                    from classical_warping import expand_parts_list as _expand
                    if set(_expand(parts_list)) >= set(cw.links_dict.keys()):
                        logger.info("All template parts selected — using full-warp (transparent canvas).")
                        parts_list = None

                people = pose_result.get("people", [])
                canvas_w = pose_result.get("canvas_width", 1024)
                canvas_h = pose_result.get("canvas_height", 1024)

                if people:
                    logger.info(f"Detected {len(people)} person(s). Generating warped silhouette...")

                    # Build canvas (replace all parts → transparent base;
                    # partial warp → start from original silhouette)
                    if parts_list:
                        final_canvas = sil_image.convert("RGBA").resize((canvas_w, canvas_h))
                    else:
                        final_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

                    for person in people:
                        person_sil = cw.generate_silhouette(
                            person,
                            (canvas_w, canvas_h),
                            parts_to_warp=parts_list,
                        )
                        final_canvas.alpha_composite(person_sil)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as warp_tmp:
                        final_canvas.save(warp_tmp, format="PNG")
                        warp_tmp_path = Path(warp_tmp.name)

                    warp_id, saved_path = SessionManager.save_silhouette(
                        warp_tmp_path, input_id, "warped_openpose"
                    )
                    warp_tmp_path.unlink()

                    stylized_resources.append(ImageResource(
                        id=warp_id,
                        url=SessionManager.get_relative_url(saved_path)
                    ))
                else:
                    logger.warning("OpenPose detected no people in the image.")

            except Exception as e:
                logger.error(f"OpenPose warping failed: {e}")
                import traceback
                traceback.print_exc()

        # Cleanup upload temp
        input_path.unlink(missing_ok=True)

        return SegmentationResponse(
            original=orig_resource,
            stylized=stylized_resources,
        )

    except Exception as e:
        logger.exception("Error in /segment")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/composite", response_model=CompositeResponse)
async def compose_image(request: CompositeRequest):
    """API 2: Silhouette ID + Background ID → random composition."""
    try:
        sil_path = SessionManager.get_silhouette_path(request.silhouette_id)
        if not sil_path:
            raise HTTPException(status_code=404, detail="Silhouette not found")

        bg_info = SessionManager.get_background_info(request.background_id)
        if not bg_info:
            raise HTTPException(status_code=404, detail="Background not found")

        bg_path = bg_info["path"]
        positions = bg_info["positions"]
        max_w = bg_info.get("max_w")
        max_h = bg_info.get("max_h")

        if not positions:
            raise HTTPException(status_code=500, detail="Background has no valid positions")

        pos = random.choice(positions)
        x, y = pos[0], pos[1]

        background = Image.open(bg_path).convert("RGBA")
        silhouette = Image.open(sil_path).convert("RGBA")

        if max_w or max_h:
            orig_w, orig_h = silhouette.size
            scale = 1.0
            if max_w and max_h:
                scale = min(max_w / orig_w, max_h / orig_h)
            elif max_w:
                scale = max_w / orig_w
            elif max_h:
                scale = max_h / orig_h
            scale = min(scale, 1.0)  # never upscale
            new_width = max(1, int(orig_w * scale))
            new_height = max(1, int(orig_h * scale))
            silhouette = silhouette.resize((new_width, new_height), Image.Resampling.LANCZOS)

        paste_x = x - (silhouette.width // 2)
        paste_y = y - (silhouette.height)

        sil_layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
        sil_layer.paste(silhouette, (paste_x, paste_y))

        final_comp = Image.alpha_composite(background, sil_layer)

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
    """API 3: Send list of image IDs to an email address."""
    try:
        paths = []
        for img_id in request.ids:
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
    """API 4: Clear all session data."""
    SessionManager.clear_session()
    return ClearResponse(status="success")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api2:app", host="0.0.0.0", port=8000, reload=True)
