"""
SOI API v3 — Hybrid Silhouette + Multi-Character Warping

Key changes from api2:
- U2Net for high-quality person silhouette (clean boundary)
- SegFormer (ATR) for body-part labels used in erasure masks
- Erasure mask clipped to U2Net alpha so boundaries are consistent
- Supports multiple characters via `base_name` parameter
- Auto-mirrors missing body-part cutouts from bilateral counterparts
- Reuses core warping functions from generate_silhouettes.py directly
"""

import sys
import json
import os
import tempfile
import shutil
import random
from pathlib import Path
from typing import List, Optional, Dict
import logging
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory so we can import generate_silhouettes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_silhouettes import (
    parse_keypoints,
    get_head_direction,
    flip_link,
    adjust_link_direction,
    get_link_transformation_matrix,
)

# Backend-local imports
from segmentation_engine import SegmentationEngine
from data_manager import SessionManager, BACKGROUNDS, BACKGROUNDS_DIR
from email_service import EmailService
from openpose_engine import OpenPoseEngine
from body_part_segmentation import BodyPartSegmenter
from image_utils import crop_to_content, tint_silhouette

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API3")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENPOSE_ROOT = Path(os.environ.get("OPENPOSE_ROOT", r"C:\openpose"))
BACKEND_DIR = Path(__file__).parent
OUTPUTS_DIR = BACKEND_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Canonical OpenPose 25-point keypoint indices for each body link.
# Used when a character's links.csv omits rows for mirrored parts.
CANONICAL_INDICES = {
    "torso":           (1,  8),
    "neck":            (1, -1),
    "head":            (-1, 0),
    "right_upper_arm": (2,  3),
    "right_forearm":   (3,  4),
    "left_upper_arm":  (5,  6),
    "left_forearm":    (6,  7),
    "right_thigh":     (9, 10),
    "right_calf":      (10, 11),
    "left_thigh":      (12, 13),
    "left_calf":       (13, 14),
    "right_foot":      (11, 22),
    "left_foot":       (14, 19),
    "right_hand":      (-1, -1),
    "left_hand":       (-1, -1),
    "hat":             (-1, -1),
}

# Bilateral mirror pairs: missing part → source part to mirror from
MIRROR_PAIRS = {
    "left_calf":       "right_calf",
    "right_calf":      "left_calf",
    "left_foot":       "right_foot",
    "right_foot":      "left_foot",
    "left_thigh":      "right_thigh",
    "right_thigh":     "left_thigh",
    "left_forearm":    "right_forearm",
    "right_forearm":   "left_forearm",
    "left_hand":       "right_hand",
    "right_hand":      "left_hand",
    "left_upper_arm":  "right_upper_arm",
    "right_upper_arm": "left_upper_arm",
}

# Body part group aliases
BODY_PART_GROUPS = {
    "arms":  ["right_upper_arm", "right_forearm", "left_upper_arm", "left_forearm"],
    "legs":  ["right_thigh", "right_calf", "left_thigh", "left_calf"],
    "feet":  ["right_foot", "left_foot"],
    "hands": ["right_hand", "left_hand"],
}

ALL_BODY_PARTS = [
    "torso", "neck", "head",
    "right_upper_arm", "right_forearm", "left_upper_arm", "left_forearm",
    "right_thigh", "right_calf", "left_thigh", "left_calf",
    "right_foot", "left_foot", "right_hand", "left_hand",
]

# Z-order for rendering (back to front)
Z_ORDER = [
    "neck",
    "left_thigh", "right_thigh", "left_calf", "right_calf",
    "left_foot", "right_foot",
    "torso",
    "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
    "head", "hat",
    "left_hand", "right_hand",
]


def expand_parts_list(parts: List[str]) -> List[str]:
    """Expand group names (like 'arms') into individual body parts."""
    expanded = []
    for p in parts:
        if p in BODY_PART_GROUPS:
            expanded.extend(BODY_PART_GROUPS[p])
        elif p in ALL_BODY_PARTS or p == "hat":
            expanded.append(p)
        else:
            logger.warning(f"Unknown body part or group: {p}")
    return list(set(expanded))


# ---------------------------------------------------------------------------
# Links loading with auto-mirroring
# ---------------------------------------------------------------------------

def load_links(links_dir: Path) -> dict:
    """
    Load body-part cutout images, reference keypoints, and direction from
    a character's links/ directory.  Missing bilateral parts are auto-
    generated by mirroring their counterpart.

    Returns:
        dict mapping link_name -> {pt1_idx, pt2_idx, pt1, pt2, image,
                                    length, direction, relative_length,
                                    mirrored: bool}
    """
    links_csv_path = links_dir / "links.csv"
    if not links_csv_path.exists():
        raise FileNotFoundError(f"links.csv not found in {links_dir}")

    # --- Phase 1: load rows from CSV that have a matching PNG ---------------
    links_df = pd.read_csv(links_csv_path)
    links_dict: Dict[str, dict] = {}
    csv_rows: Dict[str, dict] = {}  # store all CSV rows even without PNG

    for _, row in links_df.iterrows():
        name = row["name"]
        entry = {
            "pt1_idx":   int(row["pt1_idx"]),
            "pt2_idx":   int(row["pt2_idx"]),
            "pt1":       np.array([row["pt1_x"], row["pt1_y"]], dtype=float),
            "pt2":       np.array([row["pt2_x"], row["pt2_y"]], dtype=float),
            "direction": row["direction"],
        }
        csv_rows[name] = entry

        img_path = links_dir / f"{name}.png"
        if img_path.exists():
            entry["image"] = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            entry["length"] = float(np.linalg.norm(entry["pt2"] - entry["pt1"]))
            entry["mirrored"] = False
            links_dict[name] = entry
        else:
            logger.debug(f"No PNG for '{name}' in {links_dir}")

    # --- Phase 2: auto-mirror missing bilateral parts -----------------------
    for missing_name, source_name in MIRROR_PAIRS.items():
        if missing_name in links_dict:
            continue  # already loaded
        if source_name not in links_dict:
            continue  # can't mirror — source missing too

        source = links_dict[source_name]
        flipped_image = cv2.flip(source["image"], 1)  # horizontal flip
        img_w = source["image"].shape[1]

        # Flip anchor points (x → img_w - x, y unchanged)
        flipped_pt1 = np.array([img_w - source["pt1"][0], source["pt1"][1]], dtype=float)
        flipped_pt2 = np.array([img_w - source["pt2"][0], source["pt2"][1]], dtype=float)

        # Use CSV row indices if available, else fall back to canonical indices
        if missing_name in csv_rows:
            pt1_idx = csv_rows[missing_name]["pt1_idx"]
            pt2_idx = csv_rows[missing_name]["pt2_idx"]
        elif missing_name in CANONICAL_INDICES:
            pt1_idx, pt2_idx = CANONICAL_INDICES[missing_name]
        else:
            logger.warning(f"No index info for mirror target '{missing_name}', skipping")
            continue

        # Flip direction
        src_dir = source["direction"]
        flipped_dir = "l" if src_dir == "r" else "r"

        links_dict[missing_name] = {
            "pt1_idx":   pt1_idx,
            "pt2_idx":   pt2_idx,
            "pt1":       flipped_pt1,
            "pt2":       flipped_pt2,
            "image":     flipped_image,
            "length":    source["length"],
            "direction": flipped_dir,
            "mirrored":  True,
        }
        logger.info(f"Mirrored '{source_name}' -> '{missing_name}'")

    # --- Phase 3: compute relative lengths ----------------------------------
    if "torso" in links_dict:
        torso_length = links_dict["torso"]["length"]
        for ld in links_dict.values():
            ld["relative_length"] = ld["length"] / torso_length

    return links_dict


# ---------------------------------------------------------------------------
# RGBA silhouette generation (adapted from generate_silhouettes.py)
# ---------------------------------------------------------------------------

def generate_silhouette_rgba(
    person_dict: dict,
    links_dict: dict,
    canvas_size: tuple,
    parts_to_render: Optional[List[str]] = None,
    width_scale_factor: float = 0.85,
    add_hat: bool = False,
) -> Image.Image:
    """
    Generate a warped silhouette on a transparent RGBA canvas.

    Re-uses parse_keypoints / adjust_link_direction / flip_link /
    get_link_transformation_matrix from generate_silhouettes.py.
    """
    w, h = canvas_size

    # Work on a copy so flips don't persist across calls
    ld = {k: v.copy() for k, v in links_dict.items()}

    # Parse keypoints (from generate_silhouettes.py)
    kp_raw = person_dict.get("pose_keypoints_2d", [])
    keypoints = np.array(kp_raw).reshape((-1, 3)) if kp_raw else np.zeros((25, 3))

    lh_raw = person_dict.get("hand_left_keypoints_2d", [])
    left_hand = np.array(lh_raw).reshape((-1, 3)) if lh_raw else np.zeros((21, 3))

    rh_raw = person_dict.get("hand_right_keypoints_2d", [])
    right_hand = np.array(rh_raw).reshape((-1, 3)) if rh_raw else np.zeros((21, 3))

    keypoints_dict = parse_keypoints(keypoints, left_hand, right_hand, ld, add_hat)

    # Direction adjustments (from generate_silhouettes.py)
    head_dir = get_head_direction(keypoints)
    if "head" in ld and head_dir != ld["head"]["direction"]:
        ld["head"] = flip_link(ld["head"])
        if "neck" in ld:
            ld["neck"] = flip_link(ld["neck"])

    if add_hat and "hat" in ld and head_dir != ld["hat"]["direction"]:
        ld["hat"] = flip_link(ld["hat"])

    ld = adjust_link_direction(ld, keypoints_dict)

    # Determine which parts to render
    if parts_to_render is not None:
        active = set(expand_parts_list(parts_to_render))
    else:
        active = set(ld.keys())

    # Sort by Z-order
    ordered = sorted(active, key=lambda p: Z_ORDER.index(p) if p in Z_ORDER else 999)

    # Render on transparent BGRA canvas
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    for link_name in ordered:
        if link_name not in ld or link_name not in keypoints_dict:
            continue

        # Skip head when hat mode is on, and vice versa
        if link_name == "head" and add_hat:
            continue
        if link_name == "hat" and not add_hat:
            continue

        link_data = ld[link_name]
        kp_data = keypoints_dict[link_name]

        if kp_data["length"] < 1:
            continue

        src_pt1 = link_data["pt1"]
        src_pt2 = link_data["pt2"]
        dst_pt1 = kp_data["dst_pt1"]
        dst_pt2 = kp_data["dst_pt2"]

        # Width scale coefficient
        wsc = width_scale_factor
        if ("relative_length" in link_data and "relative_length" in kp_data
                and kp_data["relative_length"] > 0):
            wsc = (link_data["relative_length"]
                   / kp_data["relative_length"]
                   * width_scale_factor)

        m = get_link_transformation_matrix(src_pt1, src_pt2, dst_pt1, dst_pt2, wsc)

        warped = cv2.warpAffine(link_data["image"], m, (w, h), flags=cv2.INTER_LINEAR)

        # Alpha-composite onto canvas (BGRA)
        if warped.shape[2] == 4:
            link_alpha = warped[:, :, 3:4] / 255.0
            link_bgr = warped[:, :, :3]
            canvas_bgr = canvas[:, :, :3]
            canvas_alpha = canvas[:, :, 3:4] / 255.0

            out_alpha = link_alpha + canvas_alpha * (1 - link_alpha)
            out_alpha_safe = np.where(out_alpha == 0, 1, out_alpha)
            out_bgr = (link_bgr * link_alpha
                       + canvas_bgr * canvas_alpha * (1 - link_alpha)) / out_alpha_safe

            canvas[:, :, :3] = out_bgr.astype(np.uint8)
            canvas[:, :, 3] = (out_alpha * 255).astype(np.uint8).squeeze()

    # Convert BGRA → RGBA and return as PIL
    canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(canvas_rgba)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SOI API v3", version="3.0.0")

# Character registry: scan BACKEND_DIR for subfolders containing links/links.csv
def _list_characters() -> List[str]:
    chars = []
    for d in sorted(BACKEND_DIR.iterdir()):
        if d.is_dir() and (d / "links" / "links.csv").exists():
            chars.append(d.name)
    return chars

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
BACKGROUNDS_DIR.mkdir(exist_ok=True)
app.mount("/backgrounds", StaticFiles(directory=str(BACKGROUNDS_DIR)), name="backgrounds")

# Mount character links for frontend previews
for char_name in _list_characters():
    char_links_dir = BACKEND_DIR / char_name / "links"
    if char_links_dir.is_dir():
        app.mount(f"/assets/{char_name}", StaticFiles(directory=str(char_links_dir)), name=f"assets_{char_name}")

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_openpose_engine: Optional[OpenPoseEngine] = None
_seg_engine: Optional[SegmentationEngine] = None
_body_part_segmenter: Optional[BodyPartSegmenter] = None
_engine_cache: Dict[str, dict] = {}  # base_name -> links_dict
email_service = EmailService()


def get_openpose() -> OpenPoseEngine:
    global _openpose_engine
    if _openpose_engine is None:
        logger.info(f"Initializing OpenPose from {OPENPOSE_ROOT}")
        _openpose_engine = OpenPoseEngine(openpose_root_dir=OPENPOSE_ROOT)
    return _openpose_engine


def get_seg_engine() -> SegmentationEngine:
    """U2Net-based person silhouette extractor."""
    global _seg_engine
    if _seg_engine is None:
        logger.info("Initializing U2Net silhouette engine...")
        _seg_engine = SegmentationEngine()
    return _seg_engine


def get_segmenter() -> BodyPartSegmenter:
    """SegFormer-based body-part segmenter (ATR labels)."""
    global _body_part_segmenter
    if _body_part_segmenter is None:
        logger.info("Initializing SegFormer body-part segmenter...")
        _body_part_segmenter = BodyPartSegmenter()
    return _body_part_segmenter


def get_links(base_name: str) -> dict:
    """Load (and cache) links_dict for a character base_name."""
    if base_name not in _engine_cache:
        links_dir = BACKEND_DIR / base_name / "links"
        if not links_dir.is_dir():
            raise FileNotFoundError(f"Character directory not found: {links_dir}")
        logger.info(f"Loading links for '{base_name}' from {links_dir}")
        _engine_cache[base_name] = load_links(links_dir)
    return _engine_cache[base_name]


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
# OpenPose helper
# ---------------------------------------------------------------------------

def run_openpose_on_image(image_path: Path) -> dict:
    """Run OpenPose on a single image and return the pose result dict."""
    op = get_openpose()

    with tempfile.TemporaryDirectory() as in_tmp, \
         tempfile.TemporaryDirectory() as out_tmp:
        in_dir = Path(in_tmp)
        out_dir = Path(out_tmp)
        dest = in_dir / image_path.name
        shutil.copy(str(image_path), str(dest))

        logger.info(f"Running OpenPose on {dest.name}...")
        op.extract_pose(in_dir, out_dir)

        json_files = list(out_dir.glob("*_keypoints.json"))
        if not json_files:
            logger.warning("OpenPose produced no keypoints.")
            img = Image.open(str(image_path))
            return {"people": [], "canvas_width": img.width, "canvas_height": img.height}

        with open(json_files[0], "r") as f:
            op_data = json.load(f)

    img = Image.open(str(image_path))
    w, h = img.size

    people = []
    for p in op_data.get("people", []):
        people.append({
            "pose_keypoints_2d":       p.get("pose_keypoints_2d", []),
            "hand_left_keypoints_2d":  p.get("hand_left_keypoints_2d", []),
            "hand_right_keypoints_2d": p.get("hand_right_keypoints_2d", []),
            "face_keypoints_2d":       p.get("face_keypoints_2d", []),
        })

    return {"people": people, "canvas_width": w, "canvas_height": h}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(BACKEND_DIR / "test.html")


@app.get("/characters")
async def get_characters():
    """Return list of available character base names."""
    return _list_characters()



@app.get("/backgrounds")
async def get_backgrounds():
    bg_list = []
    for dict_key, bg_info in BACKGROUNDS.items():
        # Read bg dimensions
        w, h = 1920, 1080
        try:
            bg_path = BACKGROUNDS_DIR / bg_info["filename"]
            with Image.open(bg_path) as img:
                w, h = img.size
        except Exception as e:
            logger.warning(f"Could not read size of {bg_info['filename']}: {e}")
            
        bg_list.append({
            "id": dict_key,  # Use dictionary key directly so bg2..bg21 are unique
            "title": bg_info["filename"].split(".")[0].capitalize(),
            "url": f"/backgrounds/{bg_info['filename']}",
            "positions": bg_info.get("positions", [[w//2, h]]),
            "max_w": bg_info.get("max_w", w),
            "max_h": bg_info.get("max_h", h),
            "bg_w": w,
            "bg_h": h
        })
    return bg_list


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    image: UploadFile = File(...),
    base_name: str = Form("Prince_Achmed"),
    parts_to_warp: Optional[str] = Form(None),
    character_mapping: Optional[str] = Form(None),
):
    """
    Unified segmentation + warping endpoint.
    Uses U2Net for person silhouette, SegFormer for body-part labels.

    Args:
        image:          Uploaded photo.
        base_name:      Default character folder name.
        parts_to_warp:  Comma-separated body parts / groups to warp.
        character_mapping: JSON string mapping part groups to base_names.
                           e.g. '{"head": "Prince_AchmedSwapped", "arms": "Prince_Achmed"}'
    """
    try:
        # ------ 1. Save upload ------------------------------------------------
        suffix = Path(image.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            input_path = Path(tmp.name)

        input_id, saved_input_path = SessionManager.save_input_image(input_path)

        # ------ 2. U2Net: person silhouette -----------------------------------
        logger.info("Running U2Net for person silhouette...")
        seg_engine = get_seg_engine()
        original_photo = Image.open(str(saved_input_path)).convert("RGB")
        person_alpha_pil = seg_engine.extract_silhouette(str(saved_input_path))
        person_alpha_np = np.array(person_alpha_pil) / 255.0  # (H, W) float32 [0, 1]

        # Save original silhouette
        sil_image_cropped = crop_to_content(person_alpha_pil)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as sil_tmp:
            sil_image_cropped.save(sil_tmp, format="PNG")
            sil_tmp_path = Path(sil_tmp.name)

        orig_sil_id, orig_sil_path = SessionManager.save_silhouette(
            sil_tmp_path, input_id, "orig"
        )
        sil_tmp_path.unlink()

        orig_resource = ImageResource(
            id=orig_sil_id,
            url=SessionManager.get_relative_url(orig_sil_path),
        )

        # ------ 3. SegFormer: body-part label map -----------------------------
        logger.info("Running SegFormer for body-part labels...")
        segmenter = get_segmenter()
        label_map = segmenter.segment(original_photo)  # (H, W) uint8

        # ------ 4. Load character links (with mapping overrides) --------------
        links_dict = get_links(base_name).copy()

        if character_mapping:
            try:
                mapping = json.loads(character_mapping)
                for group, char_name in mapping.items():
                    char_links = get_links(char_name)
                    expanded = expand_parts_list([group])
                    for p in expanded:
                        if p in char_links:
                            links_dict[p] = char_links[p].copy()
            except Exception as e:
                logger.warning(f"Failed to parse or apply character_mapping: {e}")

        # ------ 5. Parse parts_to_warp ----------------------------------------
        parts_list: Optional[List[str]] = None
        if parts_to_warp:
            parts_list = [p.strip() for p in parts_to_warp.split(",")]

        # ------ 6. OpenPose -> warped silhouette(s) ---------------------------
        stylized_resources = []
        try:
            logger.info("Running OpenPose for pose detection...")
            pose_result = run_openpose_on_image(saved_input_path)

            people = pose_result.get("people", [])
            canvas_w = pose_result.get("canvas_width", 1024)
            canvas_h = pose_result.get("canvas_height", 1024)

            if people:
                logger.info(f"Detected {len(people)} person(s).")

                # Build base canvas
                if parts_list is not None:
                    # Partial warp: start with U2Net silhouette, erase warped regions
                    base_sil = person_alpha_pil.convert("RGBA").resize((canvas_w, canvas_h))
                    expanded = expand_parts_list(parts_list)

                    # Check if ALL template parts selected -> full warp
                    non_hat_parts = set(links_dict.keys()) - {"hat"}
                    if set(expanded) >= non_hat_parts:
                        logger.info("All parts selected -> full warp (transparent canvas)")
                        final_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                    else:
                        try:
                            raw_kp = people[0].get("pose_keypoints_2d", [])
                            pose_kp_arr = (np.array(raw_kp).reshape((-1, 3))
                                           if raw_kp else None)

                            erasure_mask = BodyPartSegmenter.build_erasure_mask(
                                label_map, expanded, pose_keypoints=pose_kp_arr
                            )
                            
                            final_canvas = BodyPartSegmenter.erase_parts_from_silhouette(
                                base_sil, erasure_mask
                            )
                            logger.info(
                                f"Erased {len(expanded)} body-part region(s) "
                                "from silhouette."
                            )
                        except Exception as e:
                            logger.warning(f"Body-part erasure failed: {e}")
                            final_canvas = base_sil
                else:
                    # Full warp: transparent canvas
                    final_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

                # PAD the canvas to prevent accessories like hats getting cut off out-of-bounds
                PAD = 400
                padded_w = canvas_w + 2 * PAD
                padded_h = canvas_h + 2 * PAD
                padded_canvas = Image.new("RGBA", (padded_w, padded_h), (0, 0, 0, 0))
                padded_canvas.paste(final_canvas, (PAD, PAD))
                final_canvas = padded_canvas

                # Shift OpenPose keypoints by the same padding
                for person in people:
                    kp = person.get("pose_keypoints_2d", [])
                    if kp:
                        for i in range(len(kp) // 3):
                            if kp[i * 3 + 2] > 0:  # If confidence > 0
                                kp[i * 3] += PAD      # x
                                kp[i * 3 + 1] += PAD  # y

                # Determine if hat should be added
                should_add_hat = True
                if parts_list is not None:
                    should_add_hat = "hat" in parts_list

                # Render warped parts for each detected person on padded canvas
                for person in people:
                    person_sil = generate_silhouette_rgba(
                        person, links_dict,
                        (padded_w, padded_h),
                        parts_to_render=parts_list,
                        add_hat=should_add_hat,
                    )
                    final_canvas.alpha_composite(person_sil)

                final_canvas = crop_to_content(final_canvas)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as wt:
                    final_canvas.save(wt, format="PNG")
                    warp_tmp_path = Path(wt.name)

                warp_id, saved_path = SessionManager.save_silhouette(
                    warp_tmp_path, input_id, "warped"
                )
                warp_tmp_path.unlink()

                stylized_resources.append(ImageResource(
                    id=warp_id,
                    url=SessionManager.get_relative_url(saved_path),
                ))
            else:
                logger.warning("OpenPose detected no people in the image.")

        except Exception as e:
            logger.error(f"OpenPose warping failed: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
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
    """Silhouette + background -> composite."""
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
            raise HTTPException(status_code=500, detail="No valid positions")

        pos = random.choice(positions)
        x, y = pos[0], pos[1]

        background = Image.open(bg_path).convert("RGBA")
        silhouette = Image.open(sil_path).convert("RGBA")

        silhouette_color = bg_info.get("silhouette_color")
        if silhouette_color and silhouette_color != (0, 0, 0):
            silhouette = tint_silhouette(silhouette, silhouette_color)

        if max_w or max_h:
            orig_w, orig_h = silhouette.size
            scale = 1.0
            if max_w and max_h:
                scale = min(max_w / orig_w, max_h / orig_h)
            elif max_w:
                scale = max_w / orig_w
            elif max_h:
                scale = max_h / orig_h
            scale = min(scale, 1.0)
            silhouette = silhouette.resize(
                (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))),
                Image.Resampling.LANCZOS,
            )

        paste_x = x - silhouette.width // 2
        paste_y = y - silhouette.height

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
            url=SessionManager.get_relative_url(saved_path),
        ))

    except Exception as e:
        logger.exception("Error in /composite")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/send-email", response_model=EmailResponse)
async def send_email(request: EmailRequest):
    try:
        paths = []
        for img_id in request.ids:
            path = SessionManager.find_path_by_id(img_id)
            if path:
                paths.append(path)
            else:
                logger.warning(f"Image ID not found for email: {img_id}")

        if not paths:
            return EmailResponse(status="warning", message="No valid images found.")

        result = email_service.send_email(request.email, paths)
        return EmailResponse(status="success", message=result)

    except Exception as e:
        logger.exception("Error sending email")
        return EmailResponse(status="error", message=str(e))


@app.post("/clear", response_model=ClearResponse)
async def clear_data():
    SessionManager.clear_session()
    return ClearResponse(status="success")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api3:app", host="0.0.0.0", port=8000, reload=True)
