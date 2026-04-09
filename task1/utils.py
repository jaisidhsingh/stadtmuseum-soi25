from PIL import Image
import numpy as np


def crop_to_content(image: Image.Image, margin: int = 5) -> Image.Image:
    """
    Crop an RGBA image to the bounding box of non-transparent pixels.

    Args:
        image:  PIL image (converted to RGBA if needed).
        margin: Pixel margin around the bounding box, clamped to image bounds.

    Returns:
        Cropped PIL RGBA image.  If no non-transparent pixels are found the
        original image is returned unchanged.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = np.array(image.split()[3])

    non_zero_rows = np.where(alpha.max(axis=1) > 0)[0]
    non_zero_cols = np.where(alpha.max(axis=0) > 0)[0]

    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return image

    top = max(0, int(non_zero_rows[0]) - margin)
    bottom = min(alpha.shape[0], int(non_zero_rows[-1]) + 1 + margin)
    left = max(0, int(non_zero_cols[0]) - margin)
    right = min(alpha.shape[1], int(non_zero_cols[-1]) + 1 + margin)

    return image.crop((left, top, right, bottom))