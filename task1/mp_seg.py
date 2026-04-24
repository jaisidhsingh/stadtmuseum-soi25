"""mp_seg.py

Single-pane, no-crop silhouette overlay using MediaPipe SelfieSegmentation.

Identical behaviour to single_overlay.py but the segmentation backend is
MediaPipe's SelfieSegmentation (model_selection=1) which runs in real-time
on CPU and produces soft, well-edged masks — a good middle ground between
the lightweight PPHumanSeg and the high-quality-but-slow rembg u2net models.

Usage
-----
    python mp_seg.py [--mode {rgb,sil}] [--mp-model {0,1}]

    --mp-model 0 : faster, lower quality (landscape model)
    --mp-model 1 : slightly slower, better quality (general model) [default]

Press 'q' or ESC to quit.
"""

import time
import argparse
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from utils import BG_SWITCH_INTERVAL


WIN_NAME  = " "

# Each entry: one background + its per-background anchor / scale.
# anch_y : Y coordinate (background pixels) where the *centre* of the
#          silhouette is placed.
# scale  : uniform scale applied to the full upsampled segmented frame.
DATA = [
    {"path": "assets/backgrounds_portrait/bg3_portrait.jpg",  "anch_y": 335, "scale": 0.15},
    {"path": "assets/backgrounds_portrait/bg15_portrait.jpg", "anch_y": 510, "scale": 0.15},
    {"path": "assets/backgrounds_portrait/bg11_portrait.jpg", "anch_y": 3500, "scale": 0.15},
    {"path": "assets/backgrounds_portrait/bg9_portrait.jpg",  "anch_y": 535, "scale": 0.1},
]

# Fade transition
FADE_FPS      = 60
FADE_DURATION = 0.3    # seconds per half-phase (fade-out then fade-in)

# Mask threshold — pixels with MediaPipe confidence above this are foreground.
SEG_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _build_segmenter(model_selection: int = 1):
    """Return a MediaPipe SelfieSegmentation instance."""
    return mp.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=model_selection
    )


def _segment_rgba(frame_bgr: np.ndarray, segmenter, scale: float) -> np.ndarray:
    """
    Segment *frame_bgr* with MediaPipe and return an RGBA uint8 array.

    Args:
        frame_bgr : camera frame in BGR.
        segmenter : MediaPipe SelfieSegmentation instance.
        scale     : uniform scale applied before inference (< 1 = faster).

    Returns:
        (H, W, 4) RGBA uint8 numpy array.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    result = segmenter.process(small)          # expects RGB
    # segmentation_mask: float32 (H, W), values in [0, 1]
    mask = (result.segmentation_mask > SEG_THRESHOLD).astype(np.uint8) * 255
    return np.dstack([small, mask])            # (H, W, 4) RGBA uint8


# ---------------------------------------------------------------------------
# Compositing — no crop, centre-x, anchor_y
# ---------------------------------------------------------------------------

def overlay_on_bg(frame_bgr: np.ndarray,
                  bg_bgr: np.ndarray,
                  segmenter,
                  mode: str,
                  scale: float,
                  anchor_y: int) -> np.ndarray:
    """
    Segment *frame_bgr*, scale the full RGBA result, and paste it onto
    *bg_bgr* centred horizontally with its vertical centre at *anchor_y*.

    Args:
        frame_bgr : live camera frame (BGR).
        bg_bgr    : background image (BGR, any size).
        segmenter : MediaPipe SelfieSegmentation instance.
        mode      : 'rgb' → real colours; 'sil' → dark silhouette.
        scale     : uniform scale applied to the full segmented frame.
        anchor_y  : Y coordinate of the silhouette's vertical centre (pixels).

    Returns:
        Composited BGR numpy array the same size as *bg_bgr*.
    """
    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    bh, bw = bg_bgr.shape[:2]

    # Segment → RGBA
    rgba_np = _segment_rgba(frame_bgr, segmenter, scale)
    sil_h, sil_w = rgba_np.shape[:2]

    sil_pil = Image.fromarray(rgba_np, mode="RGBA")

    # Paste position: centre-x, anchor_y as vertical centre (clamped)
    paste_x = max(0, min((bw - sil_w) // 2, bw - sil_w))
    paste_y = max(0, min(anchor_y - sil_h // 2, bh - sil_h))

    # Build full-bg RGBA canvas and alpha-composite
    fg_canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    fg_canvas.paste(sil_pil, (paste_x, paste_y))

    fg_np = np.array(fg_canvas)
    alpha = fg_np[:, :, 3] / 255.0
    m3    = alpha[:, :, np.newaxis]

    if mode == "rgb":
        fg_bgr = np.zeros((bh, bw, 3), dtype=np.uint8)
        fg_bgr[paste_y:paste_y + sil_h, paste_x:paste_x + sil_w] = (
            cv2.cvtColor(np.array(sil_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        )
        fg = fg_bgr * m3
    else:
        fg = np.zeros((bh, bw, 3), dtype=np.float64) * m3

    return (fg + bg_bgr * (1.0 - m3)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Fade transition helper
# ---------------------------------------------------------------------------

def _play_fade(win_name: str, cap, old_entry: dict, new_entry: dict,
               segmenter, mode: str) -> bool:
    """
    Fade out the current background, then fade in the next one.
    Returns False if the user pressed q/ESC during the transition.
    """
    n_frames = max(1, int(FADE_DURATION * FADE_FPS))
    wait_ms  = max(1, int(1000 / FADE_FPS))
    bh, bw   = old_entry["bg"].shape[:2]
    black    = np.zeros((bh, bw, 3), dtype=np.uint8)

    def _t(i):
        return i / (n_frames - 1) if n_frames > 1 else 1.0

    # Phase 1 — fade out to black
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        base  = overlay_on_bg(frame, old_entry["bg"], segmenter, mode,
                               old_entry["scale"], old_entry["anch_y"])
        out   = cv2.addWeighted(base, 1.0 - _t(i), black, _t(i), 0)
        cv2.imshow(win_name, out)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            return False

    # Phase 2 — fade in from black
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        base  = overlay_on_bg(frame, new_entry["bg"], segmenter, mode,
                               new_entry["scale"], new_entry["anch_y"])
        out   = cv2.addWeighted(base, _t(i), black, 1.0 - _t(i), 0)
        cv2.imshow(win_name, out)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            return False

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-pane no-crop silhouette overlay using MediaPipe segmentation."
    )
    parser.add_argument("--mode", choices=["rgb", "sil"], default="sil",
                        help="'rgb' keeps colour; 'sil' renders a dark silhouette.")
    parser.add_argument("--mp-model", type=int, choices=[0, 1], default=1,
                        help="MediaPipe model: 0=faster/lower-quality, 1=better-quality (default).")
    args = parser.parse_args()

    # Load backgrounds
    entries = []
    for d in DATA:
        bg = cv2.imread(d["path"])
        if bg is None:
            raise FileNotFoundError(f"Could not load background: {d['path']!r}")
        entries.append({"bg": bg, "scale": d["scale"], "anch_y": d["anch_y"]})

    segmenter = _build_segmenter(model_selection=args.mp_model)
    print(f"Using MediaPipe SelfieSegmentation model_selection={args.mp_model}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera capture resolution: {actual_w}x{actual_h}")

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # Size the window to the first background
    bh0, bw0 = entries[0]["bg"].shape[:2]
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, bw0, bh0)

    idx         = 0
    last_switch = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)   # mirror: left ↔ right

        cur = entries[idx]

        # Background cycling with fade
        if time.time() - last_switch >= BG_SWITCH_INTERVAL:
            next_idx = (idx + 1) % len(entries)
            # still_running = _play_fade(WIN_NAME, cap, cur, entries[next_idx],
            #                            segmenter, args.mode)
            idx         = next_idx
            last_switch = time.time()
            # if not still_running:
            #     break
            continue

        result = overlay_on_bg(frame, cur["bg"], segmenter, args.mode,
                                cur["scale"], cur["anch_y"])
        cv2.imshow(WIN_NAME, result)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    segmenter.close()


if __name__ == "__main__":
    main()
