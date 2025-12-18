import cv2 as cv
import numpy as np
from pphumanseg import PPHumanSeg

# Path to ONNX model
MODEL_PATH = "human_segmentation_pphumanseg_2023mar.onnx"

def visualize(frame, mask):
    # mask: 2D array, 0=background, 1=person
    h, w = mask.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[mask != 0] = (0, 0, 0)
    return out

def main():
    model = PPHumanSeg(modelPath=MODEL_PATH,
                       backendId=cv.dnn.DNN_BACKEND_OPENCV,
                       targetId=cv.dnn.DNN_TARGET_CPU)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        inp = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        inp = cv.resize(inp, (192, 192))
        mask = model.infer(inp)[0]
        mask = cv.resize(mask, (w, h), interpolation=cv.INTER_NEAREST)

        vis = visualize(frame, mask.astype(np.uint8))
        cv.imshow("Segmentation", vis)

        if cv.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

