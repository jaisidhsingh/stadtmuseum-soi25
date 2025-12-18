import cv2

cap = cv2.VideoCapture(0)

zoom = 0.5 

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    nh, nw = int(h/zoom), int(w/zoom)
    y1 = (h - nh)//2
    x1 = (w - nw)//2

    cropped = frame[y1:y1+nh, x1:x1+nw]
    zoomed = cv2.resize(cropped, (w, h))

    cv2.imshow("zoom", zoomed)
    if cv2.waitKey(1) == 27:
        break