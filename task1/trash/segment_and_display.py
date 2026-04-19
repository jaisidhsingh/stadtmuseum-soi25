import io
from PIL import Image
import matplotlib.pyplot as plt
from rembg import remove, new_session

from utils import crop_to_content

# 1. Initialize the rembg session
session = new_session("u2net_human_seg")

# 2. Load the source image as raw bytes (rembg works on bytes)
image_path = "man-test.jpg"
with open(image_path, "rb") as f:
    input_bytes = f.read()

# 3. Run segmentation — output is a PNG-encoded RGBA byte string
output_bytes = remove(input_bytes, session=session)

# 4. Decode the output bytes to a PIL RGBA Image
segmented_image: Image.Image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

# --- Silhouette Logic ---
# Create a silhouette: Black (0,0,0) foreground, White (255,255,255) background
alpha = segmented_image.split()[3]  # Extract the alpha channel (the mask)

# Start with a solid white image
silhouette = Image.new("RGBA", segmented_image.size, (255, 255, 255, 255))
# Create a black image to paste as foreground
black_img = Image.new("RGBA", segmented_image.size, (0, 0, 0, 255))
# Paste black onto white using the alpha channel as a mask
silhouette.paste(black_img, (0, 0), mask=alpha)

# To allow crop_to_content to work, we MUST have Alpha=0 for the "removed" parts
# currently 'silhouette' has Alpha=255 everywhere because of Image.new above.
# We put the original alpha back so the background is "transparent" for the cropper.
silhouette.putalpha(alpha)

print(f"Original size: {silhouette.size}")

# 5. Crop tightly to the content
cropped_image = crop_to_content(silhouette, margin=0)
print(f"Cropped size:  {cropped_image.size}")

# For final display, if we want the "removed" pixels to be WHITE in the viewer, 
# we can composite it over a white background or just use an RGB conversion
final_display = Image.new("RGB", cropped_image.size, (255, 255, 255))
final_display.paste(cropped_image, (0, 0), mask=cropped_image.split()[3])

# # 6. Display the result
# plt.figure(figsize=(8, 10))
# plt.imshow(final_display)
# plt.axis("off")
# plt.title("Silhouette & Cropped: man-test.jpg")
# plt.tight_layout()
# plt.show()
