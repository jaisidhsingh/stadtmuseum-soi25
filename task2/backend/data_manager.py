import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
SESSION_DIR_NAME = "session_data"
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SESSION_DIR = OUTPUTS_DIR / SESSION_DIR_NAME

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataManager")

# Backgrounds DB (Hardcoded for now)
# We expect backgrounds to be in soiBackend/backgrounds/
BACKGROUNDS_DIR = BASE_DIR / "backgrounds"
I = 1

# Cache for auto-sampled silhouette colours  (bg_id -> (R, G, B))
_sampled_color_cache: Dict[str, tuple] = {}


def _normalize_rgb(color: tuple) -> tuple:
    """Normalize RGB values to 8-bit ints for stable image APIs."""
    r, g, b = color[:3]
    r_i = int(round(float(r)))
    g_i = int(round(float(g)))
    b_i = int(round(float(b)))
    return (
        max(0, min(255, r_i)),
        max(0, min(255, g_i)),
        max(0, min(255, b_i)),
    )

BACKGROUNDS: Dict[str, Dict] = {
    # "bg1": {
    #     "id": "bg1",
    #     "filename": "bg1.jpg",
    #     "positions": [[720, 1440]],
    #     "max_w": 500 // I,
    #     "max_h": 1000 // I,
    # },
    "bg2": {
        "id": "bg2",
        "filename": "bg2.jpg",  # Reusing same file for now as placebo
        "positions": [[480, 690]],
        "max_w": 280 // I,
        "max_h": 560 // I,
        "silhouette_color": (30, 25, 20),
    },
    "bg3": {
        "id": "bg3",
        "filename": "bg3.jpg",
        "positions": [[560, 590]],
        "max_w": 180 // I,
        "max_h": 360 // I,
        "silhouette_color": (16.1, 10.2, 9.0),
    },
    "bg4": {
        "id": "bg4",
        "filename": "bg4.jpg",
        "positions": [[920, 600]],
        "max_w": 100 // I,
        "max_h": 200 // I,
        "silhouette_color": (45, 30, 20),
    },
    "bg5": {
        "id": "bg5",
        "filename": "bg5.jpg",
        "positions": [[850, 796]],
        "max_w": 402 // I,
        "max_h": 636 // I,
        "silhouette_color": (10.6, 10.2, 8.2),
    },
    # "bg6": {
    #     "id": "bg6",
    #     "filename": "bg6.png",
    #     "positions": [[185, 575]],
    #     "max_w": 122 // I,
    #     "max_h": 235 // I,
    #     "silhouette_color": (3.9, 5.1, 1.2),
    # },
    "bg7": {
        "id": "bg7",
        "filename": "bg7.png",
        "positions": [[423, 570]],
        "max_w": 149 // I,
        "max_h": 261 // I,
        "silhouette_color": (0.8, 5.9, 29.4),
    },
    "bg8": {
        "id": "bg8",
        "filename": "bg8.png",
        "positions": [[394, 468]],
        "max_w": 104 // I,
        "max_h": 153 // I,
        "silhouette_color": (5.5, 10.2, 16.9),
    },
    "bg9": {
        "id": "bg9",
        "filename": "bg9.png",
        "positions": [[663, 541]],
        "max_w": 88 // I,
        "max_h": 142 // I,
        "silhouette_color": (1.6, 3.1, 0.8),
    },
    # "bg10": {
    #     "id": "bg10",
    #     "filename": "bg10.png",
    #     "positions": [[482, 458]],
    #     "max_w": 104 // I,
    #     "max_h": 210 // I,
    #     "silhouette_color": (1.6, 9.0, 7.8),
    # },
    "bg11": {
        "id": "bg11",
        "filename": "bg11.png",
        "positions": [[324, 567]],
        "max_w": 107 // I,
        "max_h": 245 // I,
        "silhouette_color": (3.5, 1.6, 3.9),
    },
    # "bg12": {
    #     "id": "bg12",
    #     "filename": "bg12.png",
    #     "positions": [[203, 565]],
    #     "max_w": 140 // I,
    #     "max_h": 264 // I,
    #     "silhouette_color": (9.4, 6.7, 2.0),
    # },
    "bg13": {
        "id": "bg13",
        "filename": "bg13.png",
        "positions": [[1245, 1602]],
        "max_w": 375 // I,
        "max_h": 735 // I,
        "silhouette_color": (0, 0, 0),
    },
    "bg14": {
        "id": "bg14",
        "filename": "bg14.png",
        "positions": [[1194, 1827]],
        "max_w": 579 // I,
        "max_h": 1212 // I,
        "silhouette_color": (4.3, 9.0, 3.5),
    },
    "bg15": {
        "id": "bg15",
        "filename": "bg15.png",
        "positions": [[1347, 1773]],
        "max_w": 303 // I,
        "max_h": 543 // I,
        "silhouette_color": (3.1, 6.3, 2.0),
    },
    # "bg16": {
    #     "id": "bg16",
    #     "filename": "bg16.png",
    #     "positions": [[1887, 1545]],
    #     "max_w": 510 // I,
    #     "max_h": 1380 // I,
    #     "silhouette_color": (4.7, 5.1, 2.7),
    # },
    "bg17": {
        "id": "bg17",
        "filename": "bg17.png",
        "positions": [[1125, 1554]],
        "max_w": 291 // I,
        "max_h": 726 // I,
        "silhouette_color": (2.4, 7.5, 0.8),
    },
    # "bg18": {
    #     "id": "bg18",
    #     "filename": "bg18.png",
    #     "positions": [[1263, 1362]],
    #     "max_w": 504 // I,
    #     "max_h": 864 // I,
    #     "silhouette_color": (10.6, 9.4, 7.5),
    # },
    # "bg19": {
    #     "id": "bg19",
    #     "filename": "bg19.png",
    #     "positions": [[1797, 1461]],
    #     "max_w": 309 // I,
    #     "max_h": 786 // I,
    #     "silhouette_color": (13.1, 9.8, 7.8),
    # },
}


class SessionManager:
    """
    Manages file-based session storage.
    Structure:
    outputs/
      session_data/
        inputs/
        silhouettes/
        compositions/
    """

    @staticmethod
    def _ensure_dirs():
        (SESSION_DIR / "inputs").mkdir(parents=True, exist_ok=True)
        (SESSION_DIR / "silhouettes").mkdir(parents=True, exist_ok=True)
        (SESSION_DIR / "compositions").mkdir(parents=True, exist_ok=True)
        BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clear_session():
        """Deletes the entire session directory."""
        if SESSION_DIR.exists():
            shutil.rmtree(SESSION_DIR)
        SessionManager._ensure_dirs()
        logger.info("Session data cleared.")

    @staticmethod
    def save_input_image(file_path: Path) -> Tuple[str, Path]:
        """
        Saves an uploaded input image to the session input folder.
        Returns: (image_id, path_to_saved_image)
        """
        SessionManager._ensure_dirs()
        image_id = str(uuid.uuid4())
        suffix = file_path.suffix or ".png"
        filename = f"{image_id}{suffix}"
        dest_path = SESSION_DIR / "inputs" / filename
        shutil.copy2(file_path, dest_path)
        return image_id, dest_path

    @staticmethod
    def save_silhouette(
        image_path: Path, source_id: str, label: str
    ) -> Tuple[str, Path]:
        """
        Saves a generated silhouette.
        Returns: (silhouette_id, path_to_saved_image)
        """
        SessionManager._ensure_dirs()
        sil_id = f"sil_{label}_{str(uuid.uuid4())[:8]}"
        suffix = image_path.suffix or ".png"
        filename = f"{sil_id}{suffix}"
        dest_path = SESSION_DIR / "silhouettes" / filename
        shutil.copy2(image_path, dest_path)
        return sil_id, dest_path

    @staticmethod
    def save_composition(image_path: Path) -> Tuple[str, Path]:
        """
        Saves a final composite image.
        Returns: (comp_id, path_to_saved_image)
        """
        SessionManager._ensure_dirs()
        comp_id = f"comp_{str(uuid.uuid4())[:8]}"
        suffix = image_path.suffix or ".png"
        filename = f"{comp_id}{suffix}"
        dest_path = SESSION_DIR / "compositions" / filename
        shutil.copy2(image_path, dest_path)
        return comp_id, dest_path

    @staticmethod
    def get_silhouette_path(sil_id: str) -> Optional[Path]:
        """Finds a silhouette by ID."""
        for p in (SESSION_DIR / "silhouettes").glob(f"{sil_id}.*"):
            return p
        return None

    @staticmethod
    def get_background_info(bg_id: str) -> Optional[Dict]:
        """Returns background info including absolute path and silhouette colour."""
        bg = BACKGROUNDS.get(bg_id)
        if not bg:
            return None

        path = BACKGROUNDS_DIR / bg["filename"]
        if not path.exists():
            logger.warning(f"Background file not found: {path}")
            return None

        # Determine silhouette colour: manual override > cached > fresh auto-sample
        silhouette_color = bg.get("silhouette_color")
        if silhouette_color is None:
            if bg_id in _sampled_color_cache:
                silhouette_color = _sampled_color_cache[bg_id]
            else:
                from image_utils import sample_silhouette_color
                silhouette_color = sample_silhouette_color(str(path))
                _sampled_color_cache[bg_id] = silhouette_color
                logger.info(f"Auto-sampled silhouette colour for {bg_id}: {silhouette_color}")
        else:
            silhouette_color = _normalize_rgb(silhouette_color)

        return {
            "path": path,
            "positions": bg["positions"],
            "max_w": bg.get("max_w"),
            "max_h": bg.get("max_h"),
            "silhouette_color": silhouette_color,
        }

    @staticmethod
    def get_relative_url(path: Path) -> str:
        """Converts an absolute path to a URL path relative to /outputs."""
        try:
            rel = path.relative_to(OUTPUTS_DIR)
            return f"/outputs/{rel.as_posix()}"
        except ValueError:
            return ""

    @staticmethod
    def find_path_by_id(image_id: str) -> Optional[Path]:
        """
        Searches all session folders for an image with the given ID.
        """
        if not SESSION_DIR.exists():
            return None

        # Search all subfolders
        for folder in ["inputs", "silhouettes", "compositions"]:
            target_dir = SESSION_DIR / folder
            if target_dir.exists():
                # Glob for ID.* (ignoring extension)
                match = list(target_dir.glob(f"{image_id}.*"))
                if match:
                    return match[0]
        return None


# Initialize directories on module load (or first use)
SessionManager._ensure_dirs()
