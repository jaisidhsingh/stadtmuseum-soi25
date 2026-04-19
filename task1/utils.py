from PIL import Image
import cv2
import numpy as np


def crop_to_content(image: Image.Image, margin: int = 5) -> Image.Image:
    """
    Crop an RGBA image to the bounding box of non-transparent pixels.

    Args:
        image:  PIL image (converted to RGBA if needed).
        margin: Pixel margin around the bounding box, clamped to image bounds.

    Returns:
        Cropped PIL RGBA image.  If no non-transparent pixels are found the
        original image is returned unchanged.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = np.array(image.split()[3])

    non_zero_rows = np.where(alpha.max(axis=1) > 0)[0]
    non_zero_cols = np.where(alpha.max(axis=0) > 0)[0]

    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return image

    top = max(0, int(non_zero_rows[0]) - margin)
    bottom = min(alpha.shape[0], int(non_zero_rows[-1]) + 1 + margin)
    left = max(0, int(non_zero_cols[0]) - margin)
    right = min(alpha.shape[1], int(non_zero_cols[-1]) + 1 + margin)

    return image.crop((left, top, right, bottom))


BG_SWITCH_INTERVAL  = 30
TRANSITION_DURATION = 0.3
TRANSITION_FPS      = 60


def _fade_n_frames():
    return max(1, int(TRANSITION_DURATION * TRANSITION_FPS))


def _fade_wait_ms():
    return max(1, int(1000 / TRANSITION_FPS))


def play_fade_simple(win_name, cap, old_build_fn, new_build_fn, old_bg, new_bg) -> bool:
    n     = _fade_n_frames()
    wm    = _fade_wait_ms()
    h, w  = old_bg.shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)

    def _t(i):
        return i / (n - 1) if n > 1 else 1.0

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        base = old_build_fn(frame, old_bg)
        out  = cv2.addWeighted(base, 1.0 - _t(i), black, _t(i), 0)
        cv2.imshow(win_name, out)
        key = cv2.waitKey(max(1, wm)) & 0xFF
        if key in (ord("q"), 27):
            return False

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        base = new_build_fn(frame, new_bg)
        out  = cv2.addWeighted(base, _t(i), black, 1.0 - _t(i), 0)
        cv2.imshow(win_name, out)
        key = cv2.waitKey(max(1, wm)) & 0xFF
        if key in (ord("q"), 27):
            return False

    return True