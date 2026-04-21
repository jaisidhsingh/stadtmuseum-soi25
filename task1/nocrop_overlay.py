import argparse
import time
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

from utils import draw_film_reel_separator, BG_SWITCH_INTERVAL


WIN_NAME = " "

# Each entry describes one pane (square background).
# anch_y  : pane-local Y coordinate where the *centre* of the silhouette is placed.
# scale   : applied to the full upsampled segmented frame before pasting.
#
# NOTE: anchor_x is gone — the silhouette is always centred horizontally on the pane.
DATA = [
    {"path": "assets/backgrounds/bg3.jpg",  "anch_y": 460, "scale": 0.25},
    {"path": "assets/backgrounds/bg9.png",  "anch_y": 560, "scale": 0.25},
    {"path": "assets/backgrounds/bg11.png", "anch_y": 600, "scale": 0.3},
    {"path": "assets/backgrounds/bg15.png", "anch_y": 610, "scale": 0.2},
]


# ---------------------------------------------------------------------------
# Segmentation (run once per frame, shared between both panes)
# ---------------------------------------------------------------------------

def _segment_rgba(frame_bgr: np.ndarray, session) -> np.ndarray:
    """Downsample → segment → upsample back to original camera resolution."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small  = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    result = np.asarray(remove(small, session=session))   # RGBA uint8
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Per-pane compositing — no crop, centre-x paste
# ---------------------------------------------------------------------------

def _composite_on_pane(full_rgba: np.ndarray,
                       pane_bg: np.ndarray,
                       mode: str,
                       scale: float,
                       anchor_y: int) -> np.ndarray:
    """
    Scale the full upsampled RGBA frame and paste it centred horizontally
    on the pane, with its vertical centre at *anchor_y*.

    Args:
        full_rgba : RGBA numpy array at the camera's native resolution
                    (height × width × 4, uint8).  No cropping applied.
        pane_bg   : Square background (BGR, H×W×3).
        mode      : 'rgb' → real colours; 'sil' → dark silhouette.
        scale     : Uniform scale factor applied to full_rgba before pasting.
        anchor_y  : Pane-local Y coordinate of the *centre* of the silhouette.

    Returns:
        Composited BGR numpy array the same size as pane_bg.
    """
    bh, bw = pane_bg.shape[:2]

    # 1. Scale the full segmented frame
    src_h, src_w = full_rgba.shape[:2]
    sil_w = max(1, int(round(src_w * scale)))
    sil_h = max(1, int(round(src_h * scale)))

    sil_pil = Image.fromarray(full_rgba, mode="RGBA").resize(
        (sil_w, sil_h), Image.LANCZOS
    )

    # 2. Paste position: centre-x on pane, centre-y at anchor_y (clamped)
    paste_x = max(0, min((bw - sil_w) // 2, bw - sil_w))
    paste_y = max(0, min(anchor_y - sil_h // 2, bh - sil_h))

    # 3. Build full-pane RGBA canvas and alpha-composite
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
                       pane_size: int) -> np.ndarray:
    """
    Segment the camera frame once and composite it independently into pane 0
    (top) and pane 1 (bottom).  Returns a portrait canvas of shape
    (2*pane_size, pane_size, 3).
    """
    # Rotate to portrait orientation (same as double_bg.py)
    # frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Segment → full-resolution RGBA (no cropping)
    rgba_np = _segment_rgba(frame_bgr, session)

    panes = []
    for e in entries:                          # caller passes exactly 2 entries
        pane = _composite_on_pane(
            rgba_np,
            e["pane_bg"],
            mode,
            e["scale"],
            e["anch_y"],
        )
        panes.append(pane)

    canvas = np.vstack(panes)                  # (2*pane_size, pane_size, 3)
    draw_film_reel_separator(canvas, y_mid=pane_size)
    return canvas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Portrait dual-pane overlay — no silhouette crop, centre-X paste."
    )
    parser.add_argument("--mode", choices=["rgb", "sil"], default="sil",
                        help="'rgb' shows real colours; 'sil' shows a dark silhouette.")
    parser.add_argument("--pane-size", type=int, default=720,
                        help="Side length (px) of each square pane. "
                             "Window will be pane_size × (2 * pane_size).")
    args = parser.parse_args()

    pane_size = args.pane_size

    # Load and resize ALL backgrounds upfront;
    # pair 0 = entries[0:2], pair 1 = entries[2:4]
    entries = []
    for d in DATA:
        raw = cv2.imread(d["path"])
        if raw is None:
            raise FileNotFoundError(f"Could not load background: {d['path']!r}")
        pane_bg = cv2.resize(raw, (pane_size, pane_size), interpolation=cv2.INTER_AREA)
        entries.append({
            "pane_bg": pane_bg,
            "scale":   d["scale"],
            "anch_y":  d["anch_y"],
        })

    session = new_session("u2net_human_seg")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera capture resolution: {actual_w}x{actual_h}")

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
        frame = cv2.flip(frame, 1)   # mirror: left ↔ right

        # Switch pair every BG_SWITCH_INTERVAL seconds (no transition)
        if time.time() - last_switch >= BG_SWITCH_INTERVAL:
            pair_idx    = (pair_idx + 1) % 2
            last_switch = time.time()

        pair   = entries[pair_idx * 2 : pair_idx * 2 + 2]
        canvas = build_double_frame(frame, pair, session, args.mode, pane_size)
        cv2.imshow(WIN_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
