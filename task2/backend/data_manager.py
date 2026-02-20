import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

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

BACKGROUNDS: Dict[str, Dict] = {
    "bg1": {
        "id": "bg1",
        "filename": "bg1.jpg",
        "positions": [
            # [256, 700],
            # [720, 1004]
            [700, 1100]
        ],
        "scale": 1.25,
    },
    "bg2": {
        "id": "bg2",
        "filename": "bg2.jpg", # Reusing same file for now as placebo
        "positions": [
            # [350, 300]
            [350, 300]
        ],
        "scale": 0.1
    },
    "bg3": {
        "id": "bg3",
        "filename": "bg3.jpg",
        "positions": [
            # [200, 380]
            [200, 380]
        ],
        "scale": 0.2
    },
    "bg4": {
        "id": "bg4",
        "filename": "bg4.jpg",
        "positions": [
            [440, 430]
        ],
        "scale": 0.22
    },
    "bg5": {
        "id": "bg5",
        "filename": "bg5.jpg",
        "positions": [
            [440, 430]
        ],
        "scale": 0.22
    }
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
    def save_silhouette(image_path: Path, source_id: str, label: str) -> Tuple[str, Path]:
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
        """Returns background info including absolute path."""
        bg = BACKGROUNDS.get(bg_id)
        if not bg:
            return None
        
        path = BACKGROUNDS_DIR / bg["filename"]
        if not path.exists():
            logger.warning(f"Background file not found: {path}")
            return None
            
        return {
            "path": path,
            "positions": bg["positions"],
            "scale": bg.get("scale", 1.0)
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
