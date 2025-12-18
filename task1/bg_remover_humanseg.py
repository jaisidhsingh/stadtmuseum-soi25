import cv2
import numpy as np
from rembg import remove, new_session

# 1. Initialize session (Using 'u2net_human_seg' for people-only focus)
session = new_session("u2net_human_seg")

# 2. LOAD BACKGROUND IMAGE
# Replace 'background.jpg' with your actual file path.
# We load it here so we don't have to read the file every single frame.
bg_path = 'tapa-0-cleaned.jpg' 
custom_bg = cv2.imread(bg_path)

# Check if image loaded successfully
if custom_bg is None:
    print(f"Error: Could not load image at {bg_path}. Using gray background instead.")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # 3. PREPARE BACKGROUND
    # We need the background to be exactly the same size as the webcam frame.
    # We resize the loaded image to match the current frame's (width, height).
    if custom_bg is not None:
        # Resize custom image to match webcam frame size
        bg_image = cv2.resize(custom_bg, (frame.shape[1], frame.shape[0]))
    else:
        # Fallback to gray if image failed to load
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = (192, 192, 192)

    # 4. Process with RemBG (same as before)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_rgba = remove(rgb_frame, session=session)

    # Extract Alpha and normalize to 0-1
    mask = output_rgba[:, :, 3] / 255.0
    mask_3d = mask[:, :, np.newaxis]

    # 5. Composite
    # Foreground is the webcam frame, Background is the loaded image
    
    # foreground = frame * mask_3d
    white = np.full_like(frame, fill_value=1).astype(np.uint8)
    foreground = white * mask_3d
    background = bg_image * (1.0 - mask_3d)
    
    final_image = (foreground + background).astype(np.uint8)

    cv2.imshow('RemBG Custom Background', final_image)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
