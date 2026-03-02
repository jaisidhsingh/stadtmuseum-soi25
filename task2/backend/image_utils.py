"""
Image Utilities

Shared image processing functions used by both api.py and api2.py
for cropping, tinting, and color sampling.
"""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("ImageUtils")


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


def tint_silhouette(silhouette: Image.Image, color: tuple) -> Image.Image:
    """
    Replace all non-transparent pixel colours with *color*, preserving alpha.

    Args:
        silhouette: PIL RGBA image (black silhouette on transparent).
        color:      (R, G, B) target colour.

    Returns:
        New PIL RGBA image with tinted pixels.
    """
    if silhouette.mode != "RGBA":
        silhouette = silhouette.convert("RGBA")

    _, _, _, a = silhouette.split()

    return Image.merge("RGBA", (
        Image.new("L", silhouette.size, color[0]),
        Image.new("L", silhouette.size, color[1]),
        Image.new("L", silhouette.size, color[2]),
        a,
    ))


def sample_silhouette_color(image_path: str) -> tuple:
    """
    Auto-detect the dominant dark colour in a background image.

    Finds pixels where R < 60 AND G < 60 AND B < 60 and returns the
    median RGB.  Falls back to (0, 0, 0) if fewer than 10 dark pixels
    are found.

    Args:
        image_path: Path to the background image file.

    Returns:
        (R, G, B) tuple.
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)

    dark_mask = (
        (pixels[:, :, 0] < 60)
        & (pixels[:, :, 1] < 60)
        & (pixels[:, :, 2] < 60)
    )
    dark_pixels = pixels[dark_mask]

    if len(dark_pixels) < 10:
        return (0, 0, 0)

    return (
        int(np.median(dark_pixels[:, 0])),
        int(np.median(dark_pixels[:, 1])),
        int(np.median(dark_pixels[:, 2])),
    )
