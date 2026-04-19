import argparse
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

from utils import crop_to_content


BG_PATH   = "assets/tapa-0-cleaned.jpg"
SIL_SCALE = 0.4
ANCHOR_X = 450   # None → horizontally centred on the background
ANCHOR_Y = 590   # None → bottom edge of the background

WIN_NAME = " "


def _segment_rgba(frame_bgr: np.ndarray, session) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    result = np.asarray(remove(small, session=session))
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)   # H×W×4


def overlay_on_bg(frame_bgr: np.ndarray, bg_bgr: np.ndarray,
                  session, mode: str) -> np.ndarray:
    bh, bw = bg_bgr.shape[:2]

    # frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    rgba_np = _segment_rgba(frame_bgr, session)

    rgba_pil = Image.fromarray(rgba_np, mode="RGBA")
    cropped  = crop_to_content(rgba_pil)       # PIL RGBA, tight crop

    cw, ch   = cropped.size                            # tight-crop dimensions
    sil_w    = max(1, int(round(cw * SIL_SCALE)))
    sil_h    = max(1, int(round(ch * SIL_SCALE)))
    sil_pil  = cropped.resize((sil_w, sil_h), Image.LANCZOS)

    anchor_x = ANCHOR_X if ANCHOR_X is not None else bw // 2
    anchor_y = ANCHOR_Y if ANCHOR_Y is not None else bh

    paste_x = anchor_x - sil_w // 2                    # left edge
    paste_y = anchor_y - sil_h                         # top edge (bottom pinned)

    paste_x = max(0, min(paste_x, bw - sil_w))
    paste_y = max(0, min(paste_y, bh - sil_h))

    fg_canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    fg_canvas.paste(sil_pil, (paste_x, paste_y))

    fg_np  = np.array(fg_canvas)                       # H×W×4
    alpha  = fg_np[:, :, 3] / 255.0                   # float mask
    m3     = alpha[:, :, np.newaxis]

    if mode == "rgb":
        fg_bgr = np.zeros((bh, bw, 3), dtype=np.uint8)
        fg_bgr[paste_y:paste_y + sil_h, paste_x:paste_x + sil_w] = (
            cv2.cvtColor(np.array(sil_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        )
        fg = fg_bgr * m3
    else:  # "sil" — black silhouette
        fg = np.full_like(bg_bgr, fill_value=0, dtype=np.uint8) * m3

    out = (fg + bg_bgr * (1.0 - m3)).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["rgb", "sil"],
        default="sil"
    )
    args = parser.parse_args()

    bg = cv2.imread(BG_PATH)
    if bg is None:
        raise FileNotFoundError(f"Could not load background: {BG_PATH!r}")
    print(f"Background loaded: {bg.shape[1]}×{bg.shape[0]}  ({BG_PATH})")

    session = new_session("u2net_human_seg")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, bg.shape[1], bg.shape[0])
    print(f"Window: {bg.shape[1]}×{bg.shape[0]}  (press q or ESC to quit)")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        result = overlay_on_bg(frame, bg.copy(), session, args.mode)
        cv2.imshow(WIN_NAME, result)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
