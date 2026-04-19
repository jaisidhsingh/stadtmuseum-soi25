import os
import sys
import cv2
import time
import argparse
import numpy as np
import pygame
from rembg import remove, new_session
from transition_utils import play_transition, TRANSITIONS, _gray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BG_SWITCH_INTERVAL = 30            # seconds a background is displayed before switching
WIN_NAME  = "RemBG Custom Background"
BG_FOLDER = "../task2/backend/backgrounds"


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canvas helpers
# ---------------------------------------------------------------------------

def _compute_canvas_size(bg_list: list) -> tuple[int, int]:
    """Return (canvas_w, canvas_h) == dimensions of the largest background."""
    best_area = -1
    cw, ch = 640, 480  # sensible default if all images failed to load
    for bg in bg_list:
        if bg is None:
            continue
        h, w = bg.shape[:2]
        if w * h > best_area:
            best_area = w * h
            cw, ch = w, h
    return cw, ch


def _fit_to_canvas(img: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Resize *img* to fit inside the canvas preserving aspect ratio, then
    center-paste it on a black (letterbox/pillarbox) canvas."""
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = min(canvas_w / iw, canvas_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Pygame display helper
# ---------------------------------------------------------------------------

def _pg_show(screen: pygame.Surface, frame_bgr: np.ndarray) -> None:
    """Blit a BGR numpy frame to the Pygame window, scaled to fit the window."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # pygame surfarray expects axes (W, H, C) so transpose from (H, W, C)
    surface = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
    disp_w, disp_h = screen.get_size()
    if surface.get_size() != (disp_w, disp_h):
        surface = pygame.transform.smoothscale(surface, (disp_w, disp_h))
    screen.blit(surface, (0, 0))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def composite(frame_bgr: np.ndarray, bg_bgr: np.ndarray,
              seg_session, mode: str) -> np.ndarray:
    """Segment the person in *frame_bgr* and place them over *bg_bgr*.

    Both *frame_bgr* and *bg_bgr* must already be the same shape (canvas size).
    """
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.asarray(remove(rgb, session=seg_session))
    mask = rgba[:, :, 3] / 255.0
    m3   = mask[:, :, np.newaxis]
    if mode == "rgb":
        fg = frame_bgr * m3
    else:  # "sil"
        fg = np.full_like(frame_bgr, fill_value=1, dtype=np.uint8) * m3
    bg = bg_bgr * (1.0 - m3)
    return (fg + bg).astype(np.uint8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live background removal with transitions.")
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

    # --- Pygame window (fit to screen) ---
    pygame.init()
    info   = pygame.display.Info()
    scr_w  = info.current_w
    scr_h  = info.current_h
    # Use at most 90 % of the screen in each dimension
    scale  = min((scr_w * 0.90) / CANVAS_W, (scr_h * 0.90) / CANVAS_H, 1.0)
    DISP_W = max(1, int(CANVAS_W * scale))
    DISP_H = max(1, int(CANVAS_H * scale))
    print(f"Display window: {DISP_W}x{DISP_H}  (canvas: {CANVAS_W}x{CANVAS_H})")
    screen = pygame.display.set_mode((DISP_W, DISP_H))
    pygame.display.set_caption(WIN_NAME)
    clock = pygame.time.Clock()

    running = True
    while cap.isOpened() and running:
        # --- Pygame event pump (ESC / window-close to quit) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        ok, frame = cap.read()
        if not ok:
            continue

        # Scale the live camera frame to fit the canvas (preserves aspect ratio)
        frame_canvas = _fit_to_canvas(frame, CANVAS_W, CANVAS_H)

        # --- Trigger transition after BG_SWITCH_INTERVAL seconds of display ---
        if time.time() - last_bg_time >= BG_SWITCH_INTERVAL:
            next_idx    = (cur_bg_index + 1) % len(canvas_bgs)
            old_bg_safe = canvas_bgs[cur_bg_index]
            new_bg_safe = canvas_bgs[next_idx]

            play_transition(
                args.transition,
                cap, old_bg_safe, new_bg_safe,
                session, args.mode, screen,
                canvas_w=CANVAS_W, canvas_h=CANVAS_H,
            )

            cur_bg_index = next_idx
            last_bg_time = time.time()   # reset clock AFTER the transition ends
            continue

        # --- Normal render ---
        bg_image = canvas_bgs[cur_bg_index]
        final    = composite(frame_canvas, bg_image, session, args.mode)
        _pg_show(screen, final)
        clock.tick(60)   # cap at 60 fps

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
