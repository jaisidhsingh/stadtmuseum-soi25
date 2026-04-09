"""main_threaded.py
Threaded variant of main.py.

Architecture
------------
Worker thread  : cap.read() → segment → composite → push to frame_queue
Main thread    : pop from frame_queue → imshow (OpenGL-backed window for vsync)

The queue size is capped at 1 so the worker never builds a backlog of stale
frames.  If the display thread hasn't consumed the previous frame yet, the
worker discards it and pushes the newer one — this keeps latency low without
hammering the CPU.

The segmenter (rembg / u2net_human_seg) runs only as fast as the camera
delivers frames; there is no busy-wait anywhere.
"""

import os
import sys
import cv2
import time
import queue
import threading
import argparse
import numpy as np
from rembg import remove, new_session
from transition_utils import play_transition, TRANSITIONS, _gray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BG_SWITCH_INTERVAL = 5          # seconds each background is shown before switch
WIN_NAME  = "RemBG Custom Background"
BG_FOLDER = "../task2/backgrounds_cleaned"

DISPLAY_POLL_SLEEP = 0.005      # seconds the display thread sleeps when queue is empty
                                # keeps the main thread from spinning at 100 % CPU

# ---------------------------------------------------------------------------
# Shared state (written by worker, read by main thread)
# ---------------------------------------------------------------------------

# Queue of (composited_frame, timestamp_of_capture) tuples.
# maxsize=1 means the worker drops frames the display hasn't shown yet.
frame_queue: queue.Queue = queue.Queue(maxsize=1)

# Event: set by main thread to ask the worker to pause (during transitions).
pause_event = threading.Event()

# Event: set by worker to signal it is paused (main thread waits on this).
paused_event = threading.Event()

# Event: set by main thread to stop the worker on exit.
stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Compositing helper
# ---------------------------------------------------------------------------

def composite(frame_bgr: np.ndarray, bg_bgr: np.ndarray,
              seg_session, mode: str) -> np.ndarray:
    """Segment the person in *frame_bgr* and composite over *bg_bgr*."""
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
# Worker thread
# ---------------------------------------------------------------------------

def worker_loop(cap: cv2.VideoCapture, session, custom_bgs: list,
                mode: str, bg_index_ref: list) -> None:
    """Continuously capture, segment and push composited frames to frame_queue.

    *bg_index_ref* is a one-element list so the main thread can update it
    without a lock (GIL protects single-element list writes in CPython).
    """
    while not stop_event.is_set():
        # Honour pause requests from the main thread (during transitions)
        if pause_event.is_set():
            paused_event.set()
            pause_event.wait()        # block until main thread clears pause_event
            paused_event.clear()
            continue

        ok, frame = cap.read()
        if not ok:
            continue

        idx    = bg_index_ref[0]
        raw_bg = custom_bgs[idx]
        h, w   = frame.shape[:2]
        bg     = cv2.resize(raw_bg, (w, h)) if raw_bg is not None else _gray(h, w)

        result = composite(frame, bg, session, mode)

        # Replace any stale frame in the queue with the fresh one
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            frame_queue.put_nowait(result)
        except queue.Full:
            pass   # display thread grabbed it between our check and put; fine


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live background removal with threaded display."
    )
    parser.add_argument(
        "mode",
        choices=["rgb", "sil"],
        help="'rgb' keeps colour; 'sil' renders a white silhouette.",
    )
    parser.add_argument(
        "--transition",
        choices=list(TRANSITIONS),
        default="curtain",
        help=f"Transition style. One of: {list(TRANSITIONS)}. Default: curtain.",
    )
    args = parser.parse_args()

    # --- Load backgrounds ---
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

    # --- Session & camera ---
    session = new_session("u2net_human_seg")
    cap = cv2.VideoCapture(0)

    # --- OpenGL-backed window (smoother on macOS, vsync-aware) ---
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_OPENGL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Mutable bg index shared with worker
    bg_index_ref = [0]
    last_bg_time = time.time()

    # Start worker thread
    worker = threading.Thread(
        target=worker_loop,
        args=(cap, session, custom_bgs, args.mode, bg_index_ref),
        daemon=True,
    )
    worker.start()

    try:
        while True:
            # --- Transition trigger ---
            if time.time() - last_bg_time >= BG_SWITCH_INTERVAL:
                # 1. Ask the worker to pause so it doesn't fight us for cap
                pause_event.set()
                paused_event.wait(timeout=2.0)   # wait until worker is actually paused

                next_idx = (bg_index_ref[0] + 1) % len(custom_bgs)
                ok, frame = cap.read()
                if ok:
                    h, w = frame.shape[:2]
                    old_raw = custom_bgs[bg_index_ref[0]]
                    new_raw = custom_bgs[next_idx]
                    old_safe = cv2.resize(old_raw, (w, h)) if old_raw is not None else _gray(h, w)
                    new_safe = cv2.resize(new_raw, (w, h)) if new_raw is not None else _gray(h, w)

                    play_transition(
                        args.transition,
                        cap, old_safe, new_safe,
                        session, args.mode, WIN_NAME,
                    )

                bg_index_ref[0] = next_idx
                last_bg_time = time.time()

                # 2. Resume worker
                pause_event.clear()
                continue

            # --- Normal display: show whatever the worker has ready ---
            try:
                frame_out = frame_queue.get(timeout=DISPLAY_POLL_SLEEP)
                cv2.imshow(WIN_NAME, frame_out)
            except queue.Empty:
                pass   # nothing ready yet; just pump the event loop

            if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                break

    finally:
        stop_event.set()
        pause_event.clear()       # unblock worker if it was waiting
        worker.join(timeout=3.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
