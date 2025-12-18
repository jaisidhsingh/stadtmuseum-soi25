import cv2 as cv
import numpy as np
from pphumanseg import PPHumanSeg

MODEL_PATH = "human_segmentation_pphumanseg_2023mar.onnx"
BACKGROUND_PATH = "background.jpg"


def convert(result):
    h, w = result.shape
    # b_w_image = np.full((h, w, 3), 0, dtype=np.uint8)
    b_w_image = np.full((h, w, 3), 0, dtype=np.uint8)
    # b_w_image[result != 0] = [255, 255, 225]
    b_w_image[result != 0] = [1, 1, 1]
    return b_w_image 


def main():
    model = PPHumanSeg(modelPath=MODEL_PATH,
                       backendId=cv.dnn.DNN_BACKEND_OPENCV,
                       targetId=cv.dnn.DNN_TARGET_CPU)

    bg = cv.imread(BACKGROUND_PATH)
    if bg is None:
        print("Could not load background.jpg")
        return

    # convert background to grayscale → 3-channel
    bg_gray = cv.cvtColor(bg, cv.COLOR_BGR2GRAY)
    bg_gray = cv.cvtColor(bg_gray, cv.COLOR_GRAY2BGR)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        bg_resized = cv.resize(bg_gray, (w, h))

        inp = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        inp = cv.resize(inp, (192, 192))

        mask = model.infer(inp)[0]
        mask = cv.resize(mask, (w // 3, h // 3), interpolation=cv.INTER_NEAREST)
        mask3 = convert(mask)
        
        bd_mask = np.full((h, w, 3), 0, dtype=np.uint8)
        bd_mask[int(h/3) + 220 : int(2*h/3) + 220, int(w / 3) : int(2 * w / 3), :] = mask3
        
        # mask = (mask > 0.5).astype(np.uint8)

        # mask3 = mask[..., None]

        # foreground (you) from webcam, background from bw image
        out = bg_resized * (1 - bd_mask)

        cv.imshow("B/W Background + Segmented Person", out)

        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
