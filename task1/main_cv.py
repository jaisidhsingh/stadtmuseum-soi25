"""main_cv.py
Identical in behaviour to main.py but uses OpenCV (cv2.imshow) for display
instead of Pygame.  No pygame dependency is required.

Usage
-----
    python main_cv.py rgb [--transition {curtain,fade}]
    python main_cv.py sil [--transition {curtain,fade}]

Press 'q' or ESC in the OpenCV window to quit.
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from rembg import remove, new_session

# ---------------------------------------------------------------------------
# Configuration  (mirror main.py)
# ---------------------------------------------------------------------------

BG_SWITCH_INTERVAL = 30          # seconds a background is displayed before switching
WIN_NAME  = "RemBG Custom Background"
BG_FOLDER = "../task2/backend/backgrounds"

# How many pixels to shift the composited person downward on the background.
# Increase this value to push the figure further toward the bottom of the frame.
VERTICAL_OFFSET = 40

# Transition timing
TRANSITION_FPS      = 60
TRANSITION_DURATION = 0.05       # seconds per half-phase (close/open or fade-out/fade-in)

# Curtain colours (BGR)
CURTAIN_COLOR = (10, 10, 10)
FRINGE_COLOR  = (10, 10, 10)
FRINGE_H      = 15               # pixels


# ---------------------------------------------------------------------------
# Canvas helpers  (identical to main.py)
# ---------------------------------------------------------------------------

def _compute_canvas_size(bg_list: list) -> tuple[int, int]:
    """Return (canvas_w, canvas_h) == dimensions of the largest background."""
    best_area = -1
    cw, ch = 640, 480
    for bg in bg_list:
        if bg is None:
            continue
        h, w = bg.shape[:2]
        if w * h > best_area:
            best_area = w * h
            cw, ch = w, h
    return cw, ch


def _fit_to_canvas(img: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Resize *img* to fit inside the canvas preserving aspect ratio."""
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = min(canvas_w / iw, canvas_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _gray(h: int, w: int) -> np.ndarray:
    """Return a mid-grey BGR image."""
    g = np.zeros((h, w, 3), dtype=np.uint8)
    g[:] = 192
    return g


# ---------------------------------------------------------------------------
# OpenCV display helper  (replaces _pg_show from main.py)
# ---------------------------------------------------------------------------

def _cv_show(win: str, frame_bgr: np.ndarray) -> bool:
    """Display *frame_bgr* in the named OpenCV window.

    Returns False if the user pressed 'q' or ESC (quit signal).
    """
    cv2.imshow(win, frame_bgr)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)   # 27 == ESC


def _cv_delay(ms: int) -> bool:
    """Sleep for *ms* milliseconds using cv2.waitKey.

    Returns False if the user pressed 'q' or ESC during the wait.
    """
    key = cv2.waitKey(max(1, ms)) & 0xFF
    return key not in (ord("q"), 27)


# ---------------------------------------------------------------------------
# Compositing  (identical to main.py)
# ---------------------------------------------------------------------------

def composite(frame_bgr: np.ndarray, bg_bgr: np.ndarray,
              seg_session, mode: str) -> np.ndarray:
    """Segment the person in *frame_bgr* and place them over *bg_bgr*."""
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.asarray(remove(rgb, session=seg_session))
    mask = rgba[:, :, 3] / 255.0

    # --- Vertical shift: move the person downward on the background ----------
    # We create a blank mask of the same shape and paste the original mask
    # starting at row VERTICAL_OFFSET, effectively sliding the figure down.
    # Rows above the offset remain transparent (person not visible there).
    h = mask.shape[0]
    if VERTICAL_OFFSET > 0:
        shifted_mask = np.zeros_like(mask)
        # Copy only the rows that fit within the canvas after the shift
        rows_to_copy = h - VERTICAL_OFFSET
        if rows_to_copy > 0:
            shifted_mask[VERTICAL_OFFSET:, :] = mask[:rows_to_copy, :]
        mask = shifted_mask
    # -------------------------------------------------------------------------

    m3 = mask[:, :, np.newaxis]
    if mode == "rgb":
        # Apply shifted mask to the colour frame
        fg = frame_bgr * m3
    else:  # "sil"
        # Apply shifted mask to a solid white silhouette
        fg = np.full_like(frame_bgr, fill_value=1, dtype=np.uint8) * m3
    bg = bg_bgr * (1.0 - m3)
    return (fg + bg).astype(np.uint8)


# ---------------------------------------------------------------------------
# Transition helpers  (OpenCV versions — no pygame)
# ---------------------------------------------------------------------------

def _n_frames() -> int:
    return max(1, int(TRANSITION_DURATION * TRANSITION_FPS))


def _wait_ms() -> int:
    return max(1, int(1000 / TRANSITION_FPS))


def _draw_curtain_frame(base_img: np.ndarray, t: float, closing: bool) -> np.ndarray:
    """Overlay curtain panels on *base_img*.  t=0→retracted; t=1→closed."""
    h, w  = base_img.shape[:2]
    img   = base_img.copy()
    progress = t if closing else (1.0 - t)
    panel_w  = int(round(progress * w / 2))
    if panel_w == 0:
        return img

    for side in ("left", "right"):
        x0, x1 = (0, panel_w) if side == "left" else (w - panel_w, w)

        # Velvet body with subtle column shading
        panel = np.empty((h, panel_w, 3), dtype=np.float32)
        panel[:] = CURTAIN_COLOR
        cols  = np.arange(panel_w, dtype=np.float32)
        shade = 0.6 + 0.4 * (cols / max(panel_w - 1, 1))
        shade = shade[np.newaxis, :, np.newaxis]
        panel = np.clip(panel * shade, 0, 255).astype(np.uint8)

        # Faint fold lines every ~40 px
        for fx in range(0, panel_w, 40):
            lo, hi = max(0, fx - 1), min(panel_w, fx + 2)
            panel[:, lo:hi] = np.clip(
                panel[:, lo:hi].astype(np.float32) * 0.40, 0, 255
            ).astype(np.uint8)

        img[:, x0:x1] = panel

        # Fringe strip at the bottom
        fringe = np.empty((FRINGE_H, panel_w, 3), dtype=np.uint8)
        fringe[:] = FRINGE_COLOR
        for fx in range(0, panel_w, 6):
            hi_col = min(panel_w, fx + 3)
            fringe[FRINGE_H // 2:, fx:hi_col] = np.clip(
                np.array(FRINGE_COLOR, dtype=np.float32) * 0.65, 0, 255
            ).astype(np.uint8)
        img[h - FRINGE_H: h, x0:x1] = fringe

    return img


def _play_curtain(cap, old_bg, new_bg, seg_session, mode,
                  canvas_w=0, canvas_h=0) -> bool:
    """Curtain transition rendered via cv2.imshow.  Returns False if user quit."""
    n  = _n_frames()
    wm = _wait_ms()
    cw = canvas_w or old_bg.shape[1]
    ch = canvas_h or old_bg.shape[0]

    def _t(i):
        return i / (n - 1) if n > 1 else 1.0

    # Phase 1 — close
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        base = composite(_fit_to_canvas(frame, cw, ch), old_bg, seg_session, mode)
        if not _cv_show(WIN_NAME, _draw_curtain_frame(base, _t(i), closing=True)):
            return False
        _cv_delay(wm)

    # Phase 2 — open
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        base = composite(_fit_to_canvas(frame, cw, ch), new_bg, seg_session, mode)
        if not _cv_show(WIN_NAME, _draw_curtain_frame(base, _t(i), closing=False)):
            return False
        _cv_delay(wm)

    return True


def _play_fade(cap, old_bg, new_bg, seg_session, mode,
               canvas_w=0, canvas_h=0) -> bool:
    """Fade transition rendered via cv2.imshow.  Returns False if user quit."""
    n  = _n_frames()
    wm = _wait_ms()
    cw = canvas_w or old_bg.shape[1]
    ch = canvas_h or old_bg.shape[0]
    black = np.zeros((ch, cw, 3), dtype=np.uint8)

    def _t(i):
        return i / (n - 1) if n > 1 else 1.0

    # Phase 1 — fade out to black
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        alpha = 1.0 - _t(i)
        base  = composite(_fit_to_canvas(frame, cw, ch), old_bg, seg_session, mode)
        out   = cv2.addWeighted(base, alpha, black, 1.0 - alpha, 0)
        if not _cv_show(WIN_NAME, out):
            return False
        _cv_delay(wm)

    # Phase 2 — fade in from black
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        alpha = _t(i)
        base  = composite(_fit_to_canvas(frame, cw, ch), new_bg, seg_session, mode)
        out   = cv2.addWeighted(base, alpha, black, 1.0 - alpha, 0)
        if not _cv_show(WIN_NAME, out):
            return False
        _cv_delay(wm)

    return True


TRANSITIONS = {
    "curtain": _play_curtain,
    "fade":    _play_fade,
}


def play_transition(name, cap, old_bg, new_bg, seg_session, mode,
                    canvas_w=0, canvas_h=0) -> bool:
    """Dispatch to the named transition.  Returns False if the user quit."""
    fn = TRANSITIONS.get(name)
    if fn is None:
        raise ValueError(f"Unknown transition '{name}'. Choose from: {list(TRANSITIONS)}")
    return fn(cap, old_bg, new_bg, seg_session, mode,
               canvas_w=canvas_w, canvas_h=canvas_h)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live background removal with transitions (OpenCV display)."
    )
    parser.add_argument(
        "mode",
        choices=["rgb", "sil"],
        help="Render mode: 'rgb' keeps colour, 'sil' renders a white silhouette.",
    )
    parser.add_argument(
        "--transition",
        choices=list(TRANSITIONS),
        default="curtain",
        help=f"Transition style between backgrounds. One of: {list(TRANSITIONS)}. Default: curtain.",
    )
    args = parser.parse_args()

    # --- Setup ---
    session = new_session("u2net_human_seg")

    bg_paths = sorted(
        os.path.join(BG_FOLDER, f)
        for f in os.listdir(BG_FOLDER)
        if not f.startswith(".")
    )
    custom_bgs = []
    for p in bg_paths:
        bg = cv2.imread(p)
        if bg is None:
            print(f"Warning: could not load '{p}', will use grey fallback.")
        custom_bgs.append(bg)

    if not custom_bgs:
        sys.exit(f"No backgrounds found in {BG_FOLDER!r}. Aborting.")

    # --- Determine canvas size from the largest background ---
    CANVAS_W, CANVAS_H = _compute_canvas_size(custom_bgs)
    print(f"Canvas size: {CANVAS_W}x{CANVAS_H} (largest background)")

    # Pre-fit all backgrounds onto the canvas (aspect-ratio preserved)
    canvas_bgs = [
        _fit_to_canvas(bg, CANVAS_W, CANVAS_H) if bg is not None else _gray(CANVAS_H, CANVAS_W)
        for bg in custom_bgs
    ]

    cur_bg_index = 0
    last_bg_time = time.time()

    cap = cv2.VideoCapture(0)

    # --- OpenCV window (resizable so the user can drag it to any size) ---
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, CANVAS_W, CANVAS_H)
    print(f"Display window: {CANVAS_W}x{CANVAS_H}  (press q or ESC to quit)")

    running = True
    while cap.isOpened() and running:
        ok, frame = cap.read()
        if not ok:
            continue

        # Scale the live camera frame to fit the canvas
        frame_canvas = _fit_to_canvas(frame, CANVAS_W, CANVAS_H)

        # --- Trigger transition after BG_SWITCH_INTERVAL seconds ---
        if time.time() - last_bg_time >= BG_SWITCH_INTERVAL:
            next_idx    = (cur_bg_index + 1) % len(canvas_bgs)
            old_bg_safe = canvas_bgs[cur_bg_index]
            new_bg_safe = canvas_bgs[next_idx]

            still_running = play_transition(
                args.transition,
                cap, old_bg_safe, new_bg_safe,
                session, args.mode,
                canvas_w=CANVAS_W, canvas_h=CANVAS_H,
            )

            cur_bg_index = next_idx
            last_bg_time = time.time()
            if not still_running:
                break
            continue

        # --- Normal render ---
        bg_image = canvas_bgs[cur_bg_index]
        final    = composite(frame_canvas, bg_image, session, args.mode)

        if not _cv_show(WIN_NAME, final):
            break   # user pressed q / ESC

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
