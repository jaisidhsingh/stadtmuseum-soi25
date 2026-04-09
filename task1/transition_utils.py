"""transition_utils.py
Reusable background-switch transition effects for the live segmentation pipeline.

Available transitions
---------------------
- ``curtain``  : two black silhouette panels slide in from the sides (close) then retract (open).
- ``fade``     : the whole composited scene fades out to black, then fades in on the new background.

Public API
----------
``play_transition(name, cap, old_bg, new_bg, seg_session, mode, screen)``
    Dispatch to the requested transition by name.
"""

import cv2
import numpy as np
import pygame
from rembg import remove

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TRANSITION_FPS    = 60          # frames per second for all animations
TRANSITION_DURATION = 0.05        # seconds per half-phase (close / open  OR  fade-out / fade-in)

# Curtain-specific
CURTAIN_COLOR = (10, 10, 10)     # BGR: near-black
FRINGE_COLOR  = (10, 10, 10)     # BGR: matching dark fringe
FRINGE_H      = 15               # pixels


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gray(h: int, w: int) -> np.ndarray:
    """Return a mid-grey BGR image of shape (h, w, 3)."""
    g = np.zeros((h, w, 3), dtype=np.uint8)
    g[:] = 192
    return g


def _fit_to_canvas(img: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Fit *img* inside the canvas with aspect-ratio preservation (letterbox)."""
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale  = min(canvas_w / iw, canvas_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x_off = (canvas_w - nw) // 2
    y_off = (canvas_h - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def _composite(frame_bgr: np.ndarray, bg_bgr: np.ndarray,
               seg_session, mode: str) -> np.ndarray:
    """Segment *frame_bgr* and composite it over *bg_bgr*."""
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


def _n_frames() -> int:
    return max(1, int(TRANSITION_DURATION * TRANSITION_FPS))


def _wait_ms() -> int:
    return max(1, int(1000 / TRANSITION_FPS))


def _pg_show(screen: pygame.Surface, frame_bgr: np.ndarray) -> None:
    """Convert a BGR numpy frame to RGB and blit it to the Pygame window, scaled to fit."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    surface   = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
    disp_w, disp_h = screen.get_size()
    if surface.get_size() != (disp_w, disp_h):
        surface = pygame.transform.smoothscale(surface, (disp_w, disp_h))
    screen.blit(surface, (0, 0))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Curtain transition
# ---------------------------------------------------------------------------

def _draw_curtain_frame(base_img: np.ndarray, t: float, closing: bool) -> np.ndarray:
    """Overlay curtain panels on *base_img*.

    t=0 → panels fully retracted; t=1 → panels fully closed.
    When *closing* is False the animation is reversed (open phase).
    """
    h, w = base_img.shape[:2]
    img  = base_img.copy()

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
        shade = 0.6 + 0.4 * (cols / max(panel_w - 1, 1))   # 0.6 … 1.0
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


def play_curtain_transition(
    cap: cv2.VideoCapture,
    old_bg: np.ndarray,
    new_bg: np.ndarray,
    seg_session,
    mode: str,
    screen: pygame.Surface,
    canvas_w: int = 0,
    canvas_h: int = 0,
) -> None:
    """Theater-curtain: panels slide in over old bg, then retract to reveal new bg.

    *old_bg* and *new_bg* are expected to already be canvas-sized.  The live
    camera frame is also fitted to the canvas so every composited result has
    consistent dimensions.
    """
    n  = _n_frames()
    wm = _wait_ms()
    # Derive canvas dimensions from the supplied bg images when not explicit
    cw = canvas_w or old_bg.shape[1]
    ch = canvas_h or old_bg.shape[0]

    def _t(i):
        return i / (n - 1) if n > 1 else 1.0

    # Phase 1 — close
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        frame_c = _fit_to_canvas(frame, cw, ch)
        base = _composite(frame_c, old_bg, seg_session, mode)
        _pg_show(screen, _draw_curtain_frame(base, _t(i), closing=True))
        pygame.time.delay(wm)

    # Phase 2 — open
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        frame_c = _fit_to_canvas(frame, cw, ch)
        base = _composite(frame_c, new_bg, seg_session, mode)
        _pg_show(screen, _draw_curtain_frame(base, _t(i), closing=False))
        pygame.time.delay(wm)


# ---------------------------------------------------------------------------
# Fade transition
# ---------------------------------------------------------------------------

def play_fade_transition(
    cap: cv2.VideoCapture,
    old_bg: np.ndarray,
    new_bg: np.ndarray,
    seg_session,
    mode: str,
    screen: pygame.Surface,
    canvas_w: int = 0,
    canvas_h: int = 0,
) -> None:
    """Fade: composited scene fades to black over old bg, then fades in over new bg.

    *old_bg* and *new_bg* are expected to already be canvas-sized.
    """
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
        frame_c = _fit_to_canvas(frame, cw, ch)
        alpha = 1.0 - _t(i)   # 1.0 → 0.0
        base  = _composite(frame_c, old_bg, seg_session, mode)
        out   = cv2.addWeighted(base, alpha, black, 1.0 - alpha, 0)
        _pg_show(screen, out)
        pygame.time.delay(wm)

    # Phase 2 — fade in from black
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        frame_c = _fit_to_canvas(frame, cw, ch)
        alpha = _t(i)          # 0.0 → 1.0
        base  = _composite(frame_c, new_bg, seg_session, mode)
        out   = cv2.addWeighted(base, alpha, black, 1.0 - alpha, 0)
        _pg_show(screen, out)
        pygame.time.delay(wm)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TRANSITIONS = {
    "curtain": play_curtain_transition,
    "fade":    play_fade_transition,
}


def play_transition(
    name: str,
    cap: cv2.VideoCapture,
    old_bg: np.ndarray,
    new_bg: np.ndarray,
    seg_session,
    mode: str,
    screen: pygame.Surface,
    canvas_w: int = 0,
    canvas_h: int = 0,
) -> None:
    """Call the transition function identified by *name*.

    Parameters
    ----------
    name        : one of ``TRANSITIONS`` keys (e.g. ``"curtain"``, ``"fade"``)
    cap         : live VideoCapture handle
    old_bg      : current background image (BGR, canvas-sized)
    new_bg      : next background image (BGR, canvas-sized)
    seg_session : rembg session for live compositing during the animation
    mode        : ``"rgb"`` or ``"sil"``
    screen      : active Pygame display surface
    canvas_w    : canvas width (px); inferred from old_bg if 0
    canvas_h    : canvas height (px); inferred from old_bg if 0
    """
    fn = TRANSITIONS.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown transition '{name}'. Choose from: {list(TRANSITIONS)}"
        )
    fn(cap, old_bg, new_bg, seg_session, mode, screen,
       canvas_w=canvas_w, canvas_h=canvas_h)
