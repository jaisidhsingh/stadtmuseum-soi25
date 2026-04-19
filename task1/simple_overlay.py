import time
import argparse
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

from utils import crop_to_content, play_fade_simple, BG_SWITCH_INTERVAL


WIN_NAME = " "
DATA = [
    {"path": "assets/backgrounds/bg3_portrait.jpg",  "anch_x": 450, "anch_y": 590, "scale": 0.4},
    {"path": "assets/backgrounds/bg9_portrait.jpg",  "anch_x": 450, "anch_y": 590, "scale": 0.4},
    {"path": "assets/backgrounds/bg11_portrait.jpg", "anch_x": 450, "anch_y": 590, "scale": 0.4},
    {"path": "assets/backgrounds/bg15_portrait.jpg", "anch_x": 450, "anch_y": 590, "scale": 0.4},
]


def _segment_rgba(frame_bgr: np.ndarray, session) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    result = np.asarray(remove(small, session=session))
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)


def overlay_on_bg(frame_bgr: np.ndarray, bg_bgr: np.ndarray,
                  session, mode: str,
                  scale: float, anchor_x: int, anchor_y: int) -> np.ndarray:
    bh, bw = bg_bgr.shape[:2]

    rgba_np  = _segment_rgba(frame_bgr, session)
    rgba_pil = Image.fromarray(rgba_np, mode="RGBA")
    cropped  = crop_to_content(rgba_pil)

    cw, ch  = cropped.size
    sil_w   = max(1, int(round(cw * scale)))
    sil_h   = max(1, int(round(ch * scale)))
    sil_pil = cropped.resize((sil_w, sil_h), Image.LANCZOS)

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
        fg = np.full_like(bg_bgr, fill_value=0, dtype=np.uint8) * m3

    return (fg + bg_bgr * (1.0 - m3)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rgb", "sil"], default="sil")
    args = parser.parse_args()

    entries = []
    for d in DATA:
        bg = cv2.imread(d["path"])
        if bg is None:
            raise FileNotFoundError(f"Could not load background: {d['path']!r}")
        entries.append({"bg": bg, "scale": d["scale"], "anch_x": d["anch_x"], "anch_y": d["anch_y"]})

    session = new_session("u2net_human_seg")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    h0, w0 = entries[0]["bg"].shape[:2]
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, w0, h0)

    idx          = 0
    last_switch  = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        cur = entries[idx]

        if time.time() - last_switch >= BG_SWITCH_INTERVAL:
            next_idx = (idx + 1) % len(entries)
            nxt      = entries[next_idx]

            old_build = lambda frm, bg, e=cur: overlay_on_bg(frm, bg, session, args.mode, e["scale"], e["anch_x"], e["anch_y"])
            new_build = lambda frm, bg, e=nxt: overlay_on_bg(frm, bg, session, args.mode, e["scale"], e["anch_x"], e["anch_y"])

            still_running = play_fade_simple(WIN_NAME, cap, old_build, new_build, cur["bg"], nxt["bg"])
            idx         = next_idx
            last_switch = time.time()
            if not still_running:
                break
            continue

        result = overlay_on_bg(frame, cur["bg"], session, args.mode,
                               cur["scale"], cur["anch_x"], cur["anch_y"])
        cv2.imshow(WIN_NAME, result)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
