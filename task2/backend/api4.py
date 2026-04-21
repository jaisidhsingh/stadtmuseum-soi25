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
import re
import tempfile
import shutil
import random
import secrets
import zipfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import logging
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
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
OPENPOSE_ROOT = Path(os.environ.get("OPENPOSE_ROOT", r"C:\Users\lr-tech\openpose"))
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

WIDTH_SCALE_FACTORS = {
    "Achmed": 0.85,
    "Caliph": 1.0,
    "Dinarsade": 1.0,
    "Pari_Banu": 1.0
}

CANONICAL_DIRECTION = "r"


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
        direction = row["direction"]
        img_path = links_dir / f"{name}.png"
        if img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        else:
            logger.debug(f"No PNG for '{name}' in {links_dir}")
            continue
        
        pt1 = np.array([row["pt1_x"], row["pt1_y"]], dtype=float)
        pt2 = np.array([row["pt2_x"], row["pt2_y"]], dtype=float)

        if direction != CANONICAL_DIRECTION:
            # Flip direction to canonical
            img = cv2.flip(img, 1)
            img_w = img.shape[1]

            # Flip anchor points (x → img_w - x, y unchanged)
            pt1 = np.array([img_w - pt1[0], pt1[1]], dtype=float)
            pt2 = np.array([img_w - pt2[0], pt2[1]], dtype=float)

        entry = {
            "pt1_idx":   int(row["pt1_idx"]),
            "pt2_idx":   int(row["pt2_idx"]),
            "pt1":       pt1,
            "pt2":       pt2,
            "direction": CANONICAL_DIRECTION,
            "image": img,
            "length": float(np.linalg.norm(pt2 - pt1)),
            "mirrored": False
        }
        links_dict[name] = entry
        csv_rows[name] = entry

    # --- Phase 2: auto-mirror missing bilateral parts -----------------------
    for missing_name, source_name in MIRROR_PAIRS.items():
        if missing_name in links_dict:
            continue  # already loaded
        if source_name not in links_dict:
            continue  # can't mirror — source missing too

        source = links_dict[source_name]

        # Use CSV row indices if available, else fall back to canonical indices
        if missing_name in csv_rows:
            pt1_idx = csv_rows[missing_name]["pt1_idx"]
            pt2_idx = csv_rows[missing_name]["pt2_idx"]
        elif missing_name in CANONICAL_INDICES:
            pt1_idx, pt2_idx = CANONICAL_INDICES[missing_name]
        else:
            logger.warning(f"No index info for mirror target '{missing_name}', skipping")
            continue

        links_dict[missing_name] = source.copy()
        links_dict[missing_name]["pt1_idx"] = pt1_idx
        links_dict[missing_name]["pt2_idx"] = pt2_idx
        links_dict[missing_name]["mirrored"] = True
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
    cutouts_dir = BACKEND_DIR / "cutouts"
    for d in sorted(cutouts_dir.iterdir()):
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


@app.middleware("http")
async def normalize_duplicate_slashes(request: Request, call_next):
    """Normalize duplicate slashes in request path so legacy QR links keep working."""
    path = request.scope.get("path", "")
    if "//" in path:
        request.scope["path"] = re.sub(r"/{2,}", "/", path)
    return await call_next(request)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
BACKGROUNDS_DIR.mkdir(exist_ok=True)
app.mount("/backgrounds", StaticFiles(directory=str(BACKGROUNDS_DIR)), name="backgrounds")

# Mount character links for frontend previews
for char_name in _list_characters():
    char_links_dir = BACKEND_DIR / "cutouts" / char_name / "links"
    if char_links_dir.is_dir():
        app.mount(f"/assets/{char_name}", StaticFiles(directory=str(char_links_dir)), name=f"assets_{char_name}")

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_openpose_engine: Optional[OpenPoseEngine] = None
_seg_engine: Optional[SegmentationEngine] = None
_body_part_segmenter: Optional[BodyPartSegmenter] = None
_engine_cache: Dict[str, dict] = {}  # base_name -> links_dict
_stylize_runtime_cache: Dict[str, Dict[str, Any]] = {}  # context_id -> runtime artifacts
_share_tokens: Dict[str, Dict[str, Any]] = {}  # token -> {share_dir, files, expires_at, created_at}
SHARE_EXPORTS_DIR = OUTPUTS_DIR / "share_exports"
SHARE_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Privacy-first startup cleanup: remove any stale exported share files.
for stale_dir in SHARE_EXPORTS_DIR.glob("*"):
    if stale_dir.is_dir():
        shutil.rmtree(stale_dir, ignore_errors=True)


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
        preferred_dir = BACKEND_DIR / "cutouts" / base_name / "links"
        legacy_dir = BACKEND_DIR / base_name / "links"

        if preferred_dir.is_dir():
            links_dir = preferred_dir
        elif legacy_dir.is_dir():
            links_dir = legacy_dir
        else:
            raise FileNotFoundError(
                f"Character directory not found: {preferred_dir}"
            )

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
    context_id: Optional[str] = None


class StylizeRequest(BaseModel):
    context_id: str
    base_name: str = "Prince_Achmed"
    parts_to_warp: Optional[str] = None
    character_mapping: Optional[Dict[str, str]] = None


class StylizeResponse(BaseModel):
    stylized: List[ImageResource]

class CompositeRequest(BaseModel):
    silhouette_id: str
    background_id: str

class CompositeResponse(BaseModel):
    result: ImageResource

class ClearResponse(BaseModel):
    status: str


class ShareCreateRequest(BaseModel):
    ids: List[str]
    ttl_minutes: Optional[int] = 30


class ShareCreateResponse(BaseModel):
    token: str
    share_url: str
    expires_at: str


class ShareInfoResponse(BaseModel):
    token: str
    urls: List[str]
    expires_at: str


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


def _resolve_context_input_path(context_id: str) -> Path:
    p = SessionManager.find_path_by_id(context_id)
    if not p or not p.exists():
        raise FileNotFoundError(f"Context not found: {context_id}")
    if "inputs" not in p.parts:
        raise FileNotFoundError(f"Context does not reference an input image: {context_id}")
    return p


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _cleanup_expired_share_tokens() -> None:
    now = _now_utc()
    expired = [
        token for token, entry in _share_tokens.items()
        if entry.get("expires_at") <= now
    ]
    for token in expired:
        entry = _share_tokens.pop(token, None)
        if not entry:
            continue
        share_dir = entry.get("share_dir")
        if share_dir:
            shutil.rmtree(share_dir, ignore_errors=True)


def _public_base_url() -> str:
    # Set this in production/tunnel mode, e.g. https://soi.yourdomain.com
    default_public_url = "https://earrings-dale-sport-patterns.trycloudflare.com"
    return os.environ.get("PUBLIC_BASE_URL", default_public_url).rstrip("/")


def _build_share_url(token: str) -> str:
    return f"{_public_base_url()}/share/{token}"


def _create_share_token(ids: List[str], ttl_minutes: int) -> ShareCreateResponse:
    _cleanup_expired_share_tokens()

    if not ids:
        raise HTTPException(status_code=400, detail="No image ids provided")

    share_files: List[str] = []
    share_id = secrets.token_hex(8)
    share_dir = SHARE_EXPORTS_DIR / share_id
    share_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_id in enumerate(ids, start=1):
        path = SessionManager.find_path_by_id(image_id)
        if not path:
            continue
        # Share endpoint is intended for final rendered compositions only.
        if "compositions" not in path.parts:
            continue
        filename = f"image_{idx:02d}.png"
        dst = share_dir / filename
        try:
            shutil.copy2(path, dst)
            share_files.append(filename)
        except Exception as e:
            logger.warning(f"Failed to copy shared image {image_id}: {e}")

    if not share_files:
        shutil.rmtree(share_dir, ignore_errors=True)
        raise HTTPException(status_code=404, detail="No valid images found for sharing")

    ttl = max(1, min(180, int(ttl_minutes)))
    created_at = _now_utc()
    expires_at = created_at + timedelta(minutes=ttl)

    token = secrets.token_urlsafe(16)
    _share_tokens[token] = {
        "share_dir": str(share_dir),
        "files": share_files,
        "created_at": created_at,
        "expires_at": expires_at,
    }

    return ShareCreateResponse(
        token=token,
        share_url=_build_share_url(token),
        expires_at=expires_at.isoformat(),
    )


def _get_share_entry(token: str) -> Dict[str, Any]:
    _cleanup_expired_share_tokens()
    entry = _share_tokens.get(token)
    if not entry:
        raise HTTPException(status_code=404, detail="Share link not found or expired")
    return entry


def _resolve_share_paths(entry: Dict[str, Any]) -> List[Path]:
    share_dir_raw = entry.get("share_dir")
    files = entry.get("files", [])
    if not share_dir_raw or not files:
        return []

    share_dir = Path(share_dir_raw)
    paths: List[Path] = []
    for name in files:
        p = (share_dir / str(name)).resolve()
        try:
            p.relative_to(share_dir.resolve())
        except Exception:
            continue
        if p.exists() and p.is_file():
            paths.append(p)
    return paths


def _share_image_url(token: str, index: int) -> str:
    return f"{_public_base_url()}/share/{token}/image/{index}"


def _stylize_from_context(
    context_id: str,
    base_name: str,
    parts_to_warp: Optional[str],
    character_mapping: Optional[Dict[str, str]],
) -> List[ImageResource]:
    input_path = _resolve_context_input_path(context_id)

    runtime = _stylize_runtime_cache.setdefault(context_id, {})

    if "original_photo" not in runtime:
        runtime["original_photo"] = Image.open(str(input_path)).convert("RGB")
    original_photo: Image.Image = runtime["original_photo"]

    if "person_alpha" not in runtime:
        logger.info(f"[stylize:{context_id}] Running U2Net person silhouette...")
        runtime["person_alpha"] = get_seg_engine().extract_silhouette(str(input_path))
    person_alpha_pil: Image.Image = runtime["person_alpha"]

    if "label_map" not in runtime:
        logger.info(f"[stylize:{context_id}] Running SegFormer body-part labels (first stylize call)...")
        runtime["label_map"] = get_segmenter().segment(original_photo)
    label_map: np.ndarray = runtime["label_map"]

    if "pose_result" not in runtime:
        logger.info(f"[stylize:{context_id}] Running OpenPose (first stylize call)...")
        runtime["pose_result"] = run_openpose_on_image(input_path)
    pose_result: dict = runtime["pose_result"]

    if "people_shifted" not in runtime:
        people = pose_result.get("people", [])
        pad = 400
        shifted: List[dict] = []
        for person in people:
            p2 = {
                "pose_keypoints_2d": list(person.get("pose_keypoints_2d", [])),
                "hand_left_keypoints_2d": list(person.get("hand_left_keypoints_2d", [])),
                "hand_right_keypoints_2d": list(person.get("hand_right_keypoints_2d", [])),
                "face_keypoints_2d": list(person.get("face_keypoints_2d", [])),
            }
            kp = p2.get("pose_keypoints_2d", [])
            if kp:
                for i in range(len(kp) // 3):
                    if kp[i * 3 + 2] > 0:
                        kp[i * 3] += pad
                        kp[i * 3 + 1] += pad
            shifted.append(p2)
        runtime["people_shifted"] = shifted

    if "pose_kp_arr" not in runtime:
        people = pose_result.get("people", [])
        raw_kp = people[0].get("pose_keypoints_2d", []) if people else []
        runtime["pose_kp_arr"] = np.array(raw_kp).reshape((-1, 3)) if raw_kp else None

    links_dict = get_links(base_name).copy()
    if character_mapping:
        try:
            for group, char_name in character_mapping.items():
                char_links = get_links(char_name)
                expanded = expand_parts_list([group])
                for p in expanded:
                    if p in char_links:
                        links_dict[p] = char_links[p].copy()
        except Exception as e:
            logger.warning(f"Failed to apply character_mapping for stylize: {e}")

    parts_list: Optional[List[str]] = None
    if parts_to_warp:
        parts_list = [p.strip() for p in parts_to_warp.split(",") if p.strip()]

    stylized_resources: List[ImageResource] = []

    people = pose_result.get("people", [])
    canvas_w = pose_result.get("canvas_width", 1024)
    canvas_h = pose_result.get("canvas_height", 1024)

    if not people:
        logger.warning(f"[stylize:{context_id}] OpenPose detected no people.")
        return stylized_resources

    if "base_sil" not in runtime:
        runtime["base_sil"] = person_alpha_pil.convert("RGBA").resize((canvas_w, canvas_h))
    base_sil = runtime["base_sil"]

    # if parts_list is not None:
    #     expanded = expand_parts_list(parts_list)

    #     non_hat_parts = set(links_dict.keys()) - {"hat"}
    #     if set(expanded) >= non_hat_parts:
    #         final_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    #     else:
    #         erasure_mask = BodyPartSegmenter.build_erasure_mask(
    #             label_map,
    #             expanded,
    #             pose_keypoints=runtime.get("pose_kp_arr"),
    #         )
    #         final_canvas = BodyPartSegmenter.erase_parts_from_silhouette(base_sil, erasure_mask)
    # else:
    #     final_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    final_canvas = base_sil.copy()

    pad = 400
    padded_w = canvas_w + 2 * pad
    padded_h = canvas_h + 2 * pad
    padded_canvas = Image.new("RGBA", (padded_w, padded_h), (0, 0, 0, 0))
    padded_canvas.paste(final_canvas, (pad, pad))
    final_canvas = padded_canvas

    people_shifted = runtime.get("people_shifted", [])

    should_add_hat = True
    if parts_list is not None:
        should_add_hat = "hat" in parts_list

    for person in people_shifted:
        person_sil = generate_silhouette_rgba(
            person,
            links_dict,
            (padded_w, padded_h),
            parts_to_render=parts_list,
            width_scale_factor=WIDTH_SCALE_FACTORS.get(base_name, 1.0),
            add_hat=should_add_hat,
        )
        final_canvas.alpha_composite(person_sil)

    final_canvas = crop_to_content(final_canvas)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as wt:
        final_canvas.save(wt, format="PNG")
        warp_tmp_path = Path(wt.name)

    warp_id, saved_path = SessionManager.save_silhouette(warp_tmp_path, context_id, "warped")
    warp_tmp_path.unlink()

    stylized_resources.append(
        ImageResource(id=warp_id, url=SessionManager.get_relative_url(saved_path))
    )
    return stylized_resources


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
        bg_path = BACKGROUNDS_DIR / bg_info["filename"]
        if not bg_path.exists():
            logger.warning(f"Skipping missing background file: {bg_info['filename']}")
            continue

        # Read bg dimensions
        w, h = 1920, 1080
        try:
            with Image.open(bg_path) as img:
                w, h = img.size
        except Exception as e:
            logger.warning(f"Could not read size of {bg_info['filename']}: {e}")

        raw_color = bg_info.get("silhouette_color", (0, 0, 0))
        if raw_color is None:
            raw_color = (0, 0, 0)
        silhouette_color = [
            max(0, min(255, int(round(float(raw_color[0]))))),
            max(0, min(255, int(round(float(raw_color[1]))))),
            max(0, min(255, int(round(float(raw_color[2]))))),
        ]
            
        bg_list.append({
            "id": dict_key,  # Use dictionary key directly so bg2..bg21 are unique
            "title": bg_info["title"],
            "url": f"/backgrounds/{bg_info['filename']}",
            "positions": bg_info.get("positions", [[w//2, h]]),
            "max_w": bg_info.get("max_w", w),
            "max_h": bg_info.get("max_h", h),
            "bg_w": w,
            "bg_h": h,
            "silhouette_color": silhouette_color,
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
    Silhouette generation endpoint.
    This endpoint only generates the original silhouette and creates a context_id
    for subsequent stylization calls.

    Args:
        image:          Uploaded photo.
        base_name, parts_to_warp, character_mapping are accepted for backward
        compatibility but ignored by this endpoint.
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

        # Cleanup
        input_path.unlink(missing_ok=True)

        # Prime stylize runtime with already-computed silhouette data so first
        # stylize call skips a redundant U2Net pass.
        runtime = _stylize_runtime_cache.setdefault(input_id, {})
        runtime["original_photo"] = original_photo
        runtime["person_alpha"] = person_alpha_pil

        # Silhouette-only endpoint: stylization is handled by /stylize.
        stylized_resources: List[ImageResource] = []

        return SegmentationResponse(
            original=orig_resource,
            stylized=stylized_resources,
            context_id=input_id,
        )

    except Exception as e:
        logger.exception("Error in /segment")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stylize", response_model=StylizeResponse)
async def stylize_image(request: StylizeRequest):
    """
    Stylization endpoint.
    Uses a previously generated context_id from /segment and performs warping.
    SegFormer/OpenPose/U2Net are computed once per context and reused for
    subsequent stylization updates in the same session.
    """
    try:
        stylized = _stylize_from_context(
            context_id=request.context_id,
            base_name=request.base_name,
            parts_to_warp=request.parts_to_warp,
            character_mapping=request.character_mapping,
        )
        return StylizeResponse(stylized=stylized)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error in /stylize")
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


@app.post("/share/create", response_model=ShareCreateResponse)
async def create_share_link(request: ShareCreateRequest):
    return _create_share_token(request.ids, request.ttl_minutes or 30)


@app.get("/share/info/{token}", response_model=ShareInfoResponse)
async def get_share_info(token: str):
    entry = _get_share_entry(token)
    paths = _resolve_share_paths(entry)
    urls = [_share_image_url(token, i) for i in range(len(paths))]
    return ShareInfoResponse(
        token=token,
        urls=urls,
        expires_at=entry["expires_at"].isoformat(),
    )


@app.get("/share/{token}/image/{index}")
async def get_shared_image(token: str, index: int):
    entry = _get_share_entry(token)
    paths = _resolve_share_paths(entry)

    if index < 0 or index >= len(paths):
        raise HTTPException(status_code=404, detail="Shared image not found")

    return FileResponse(paths[index])


@app.get("/share/{token}/download/{index}")
async def download_shared_image(token: str, index: int):
    entry = _get_share_entry(token)
    paths = _resolve_share_paths(entry)

    if index < 0 or index >= len(paths):
        raise HTTPException(status_code=404, detail="Shared image not found")

    filename = f"shared-image-{index + 1}.png"
    return FileResponse(paths[index], media_type="image/png", filename=filename)


@app.get("/share/{token}/download-all")
async def download_all_shared_images(token: str):
    entry = _get_share_entry(token)
    paths = _resolve_share_paths(entry)

    if not paths:
        raise HTTPException(status_code=404, detail="No shared images available")

    share_dir = Path(entry.get("share_dir", ""))
    if not share_dir:
        raise HTTPException(status_code=404, detail="Share export folder missing")

    zip_path = share_dir / "all-images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, image_path in enumerate(paths, start=1):
            zf.write(image_path, arcname=f"shared-image-{i}.png")

    return FileResponse(zip_path, media_type="application/zip", filename="shared-images.zip")


@app.get("/share/{token}", response_class=HTMLResponse)
async def open_share_page(token: str):
    entry = _get_share_entry(token)
    expires = entry["expires_at"].strftime("%Y-%m-%d %H:%M:%S UTC")
    paths = _resolve_share_paths(entry)

    if not paths:
        raise HTTPException(status_code=404, detail="No shared images available")

    items_html = "".join(
        f'''
<div class="item-card">
    <label class="pick-row">
        <input type="checkbox" class="pick-checkbox" data-index="{i}" checked />
        <span>Select image {i + 1}</span>
    </label>
    <img src="/share/{token}/image/{i}" alt="Shared image {i + 1}" class="share-image" />
    <div class="item-actions">
        <a class="btn btn-download" href="/share/{token}/download/{i}">Download</a>
        <button class="btn btn-share" onclick="shareImage('/share/{token}/image/{i}', {i + 1})">Share</button>
    </div>
</div>
'''
        for i in range(len(paths))
    )

    html_template = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Session Share</title>
    <style>
        :root {
            color-scheme: light;
        }
        body {
            font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            max-width: 980px;
            margin: 24px auto;
            padding: 0 16px 24px;
            background: #f8f8f6;
            color: #1d1d1b;
        }
        .item-card {
            margin: 16px 0;
            padding: 12px;
            background: #ffffff;
            border: 1px solid #e8e6df;
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .share-image {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #ddd;
            display: block;
        }
        .bulk-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 12px 0 16px;
        }
        .pick-row {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            font-size: 14px;
            color: #333;
        }
        .pick-checkbox {
            width: 18px;
            height: 18px;
        }
        .item-actions {
            margin-top: 10px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            appearance: none;
            border: none;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 15px;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            line-height: 1;
        }
        .btn-download {
            background: #1d4ed8;
            color: #fff;
        }
        .btn-download-all {
            background: #2563eb;
            color: #fff;
        }
        .btn-download-selected {
            background: #0f766e;
            color: #fff;
        }
        .btn-share {
            background: #111827;
            color: #fff;
        }
        .btn-share-selected {
            background: #7c3aed;
            color: #fff;
        }
        .hint {
            color: #5a5a58;
            font-size: 14px;
        }
    </style>
</head>
<body>
  <h1>Shared Session Images</h1>
  <p>This link expires at <strong>__EXPIRES__</strong>.</p>
    <p class=\"hint\">Use Download to save an image, or Share to open your phone's share options.</p>
  <div class="bulk-actions">
    <button class="btn" onclick="setAllSelected(true)">Select All</button>
    <button class="btn" onclick="setAllSelected(false)">Clear Selection</button>
    <a class="btn btn-download-all" href="/share/__TOKEN__/download-all">Download All (ZIP)</a>
    <button class="btn btn-download-selected" onclick="downloadSelected()">Download Selected</button>
    <button class="btn btn-share-selected" onclick="shareSelected()">Share Selected</button>
  </div>
  __ITEMS_HTML__
    <script>
        function selectedIndexes() {
            return Array.from(document.querySelectorAll('.pick-checkbox:checked'))
                .map((el) => Number(el.dataset.index))
                .filter((n) => Number.isInteger(n) && n >= 0);
        }

        function setAllSelected(checked) {
            document.querySelectorAll('.pick-checkbox').forEach((el) => {
                el.checked = checked;
            });
        }

        async function shareFilesOrLinks(items) {
            if (!items.length) {
                alert('Select at least one image first.');
                return;
            }

            if (navigator.share) {
                // Try sharing actual files first.
                try {
                    const files = [];
                    for (const item of items) {
                        const res = await fetch(item.absoluteUrl);
                        if (!res.ok) continue;
                        const blob = await res.blob();
                        files.push(new File([blob], item.filename, { type: blob.type || 'image/png' }));
                    }

                    if (files.length && navigator.canShare && navigator.canShare({ files })) {
                        await navigator.share({
                            title: files.length === 1 ? 'Shared image' : 'Shared images',
                            text: files.length === 1 ? 'Shared image' : 'Shared images',
                            files,
                        });
                        return;
                    }
                } catch (fileShareErr) {
                    console.warn('Selected file share fallback:', fileShareErr);
                }

                // Fallback to sharing first selected URL.
                const first = items[0];
                await navigator.share({ title: 'Shared image', text: 'Shared image', url: first.absoluteUrl });
                return;
            }

            const urls = items.map((x) => x.absoluteUrl).join('\n');
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(urls);
                alert('Share not supported on this browser. Selected links copied to clipboard.');
                return;
            }

            prompt('Copy these links to share:', urls);
        }

        async function shareImage(relativeUrl, index) {
            const absoluteUrl = new URL(relativeUrl, window.location.origin).toString();
            try {
                await shareFilesOrLinks([
                    {
                        absoluteUrl,
                        filename: 'shared-image-' + index + '.png',
                    }
                ]);
            } catch (err) {
                // User-cancelled share actions should stay quiet.
                if (!err || err.name !== 'AbortError') {
                    console.error(err);
                    alert('Could not open sharing options on this device.');
                }
            }
        }

        function downloadSelected() {
            const indexes = selectedIndexes();
            if (!indexes.length) {
                alert('Select at least one image first.');
                return;
            }

            indexes.forEach((i) => {
                const a = document.createElement('a');
                a.href = '/share/__TOKEN__/download/' + i;
                a.rel = 'noopener';
                document.body.appendChild(a);
                a.click();
                a.remove();
            });
        }

        async function shareSelected() {
            const indexes = selectedIndexes();
            if (!indexes.length) {
                alert('Select at least one image first.');
                return;
            }

            const items = indexes.map((i) => ({
                absoluteUrl: new URL('/share/__TOKEN__/image/' + i, window.location.origin).toString(),
                filename: 'shared-image-' + (i + 1) + '.png',
            }));

            try {
                await shareFilesOrLinks(items);
            } catch (err) {
                if (!err || err.name !== 'AbortError') {
                    console.error(err);
                    alert('Could not open sharing options on this device.');
                }
            }
        }
    </script>
</body>
</html>
"""
    html = (
        html_template
        .replace("__EXPIRES__", expires)
        .replace("__ITEMS_HTML__", items_html)
        .replace("__TOKEN__", token)
    )
    return HTMLResponse(content=html)


@app.post("/clear", response_model=ClearResponse)
async def clear_data():
    _stylize_runtime_cache.clear()
    for entry in _share_tokens.values():
        share_dir = entry.get("share_dir")
        if share_dir:
            shutil.rmtree(share_dir, ignore_errors=True)
    _share_tokens.clear()
    SessionManager.clear_session()
    return ClearResponse(status="success")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api3:app", host="0.0.0.0", port=8000, reload=True)
