import argparse
import time
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

from utils import (crop_to_content, draw_film_reel_separator, BG_SWITCH_INTERVAL)


WIN_NAME = " "

# Each entry describes one pane (square background).
# anch_x / anch_y are in pane-local pixel coordinates.
# scale  is applied to DATA-space dimensions before pasting.
DATA = [
    {"path": "assets/backgrounds/bg11.png", "anch_x": 360, "anch_y": 710, "scale": 0.3},
    {"path": "assets/backgrounds/bg15.png", "anch_x": 500, "anch_y": 690, "scale": 0.2},
    {"path": "assets/backgrounds/bg3.jpg",  "anch_x": 550, "anch_y": 560, "scale": 0.25},
    {"path": "assets/backgrounds/bg9.png",  "anch_x": 720, "anch_y": 660, "scale": 0.2},
]


# ---------------------------------------------------------------------------
# Segmentation (run once per frame, shared between both panes)
# ---------------------------------------------------------------------------

def _segment_rgba(frame_bgr: np.ndarray, session) -> np.ndarray:
    """Downsample → segment → upsample back to original camera resolution."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    result = np.asarray(remove(small, session=session))          # RGBA uint8
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Per-pane compositing (identical logic to simple_overlay.py)
# ---------------------------------------------------------------------------

def _composite_on_pane(cropped_rgba: Image.Image,
                       pane_bg: np.ndarray,
                       mode: str,
                       scale: float,
                       anchor_x: int,
                       anchor_y: int) -> np.ndarray:
    """
    Paste the pre-cropped RGBA silhouette onto a square pane background.

    Args:
        cropped_rgba : content-cropped RGBA PIL image (already segmented).
        pane_bg      : square background numpy array (BGR, H×W×3).
        mode         : 'rgb' → show real colours; 'sil' → dark silhouette.
        scale        : silhouette scale factor (DATA space).
        anchor_x     : pane-local horizontal centre of the paste (pixels).
        anchor_y     : pane-local bottom edge of the paste (pixels).

    Returns:
        Composited BGR numpy array the same size as pane_bg.
    """
    bh, bw = pane_bg.shape[:2]

    cw, ch  = cropped_rgba.size
    sil_w   = max(1, int(round(cw * scale)))
    sil_h   = max(1, int(round(ch * scale)))
    sil_pil = cropped_rgba.resize((sil_w, sil_h), Image.LANCZOS)

    paste_x = max(0, min(anchor_x - sil_w // 2, bw - sil_w))
    paste_y = max(0, min(anchor_y - sil_h,       bh - sil_h))

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

    return (fg + pane_bg * (1.0 - m3)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame builder: segment once, composite into two panes, stack vertically
# ---------------------------------------------------------------------------

def build_double_frame(frame_bgr: np.ndarray,
                       entries: list,
                       session,
                       mode: str,
                       pane_size: int,
                       reel_strip: np.ndarray) -> np.ndarray:
    """
    Segment the frame once and composite it independently into pane 0 (top)
    and pane 1 (bottom).  Returns a portrait canvas of shape (2*pane_size,
    pane_size, 3).
    """
    # --- segment once ---
    rgba_np  = _segment_rgba(frame_bgr, session)
    rgba_pil = Image.fromarray(rgba_np, mode="RGBA")
    cropped  = crop_to_content(rgba_pil)           # tight crop (no wasted space)

    panes = []
    for e in entries:                              # caller passes exactly 2 entries
        pane   = _composite_on_pane(
            cropped,
            e["pane_bg"],
            mode,
            e["scale"],
            e["anch_x"],
            e["anch_y"],
        )
        panes.append(pane)

    canvas = np.vstack(panes)                      # (2*pane_size, pane_size, 3)
    draw_film_reel_separator(canvas, y_mid=pane_size+0)
    return canvas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Portrait dual-pane background overlay."
    )
    parser.add_argument("--mode", choices=["rgb", "sil"], default="sil",
                        help="'rgb' shows real colours; 'sil' shows a dark silhouette.")
    parser.add_argument("--pane-size", type=int, default=720,
                        help="Side length (px) of each square pane. "
                             "Window will be pane_size × (2 * pane_size).")
    args = parser.parse_args()

    pane_size = args.pane_size

    # Load and resize ALL backgrounds upfront; pair 0 = entries[0:2], pair 1 = entries[2:4]
    entries = []
    for d in DATA:
        raw = cv2.imread(d["path"])
        if raw is None:
            raise FileNotFoundError(f"Could not load background: {d['path']!r}")
        pane_bg = cv2.resize(raw, (pane_size, pane_size), interpolation=cv2.INTER_AREA)
        entries.append({
            "pane_bg": pane_bg,
            "scale":   d["scale"],
            "anch_x":  d["anch_x"],
            "anch_y":  d["anch_y"],
        })

    # Bake the reel separator strip once (scales to pane width)
    reel_strip = None

    session = new_session("u2net_human_seg")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # Portrait window: width = pane_size, height = 2 * pane_size
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, pane_size, 2 * pane_size)

    # Background-pair cycling: pair 0 → entries[0:2], pair 1 → entries[2:4]
    pair_idx    = 0
    last_switch = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        # Switch pair every BG_SWITCH_INTERVAL seconds (no transition)
        if time.time() - last_switch >= BG_SWITCH_INTERVAL:
            pair_idx    = (pair_idx + 1) % 2
            last_switch = time.time()

        pair = entries[pair_idx * 2 : pair_idx * 2 + 2]
        canvas = build_double_frame(frame, pair, session, args.mode, pane_size, reel_strip)
        cv2.imshow(WIN_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
