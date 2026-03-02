"""
Body-Part Semantic Segmentation

Uses a SegFormer model fine-tuned on the ATR dataset to produce per-pixel
body-part labels.  The resulting label map is used to build erasure masks
that remove specific body regions from the original silhouette before
warped template parts are composited on top.
"""

import logging
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

logger = logging.getLogger("BodyPartSegmenter")

# ---------------------------------------------------------------------------
# ATR label definitions (mattmdjaga/segformer_b2_clothes)
# ---------------------------------------------------------------------------
#  0: Background   1: Hat          2: Hair        3: Sunglasses
#  4: Upper-clothes 5: Skirt       6: Pants       7: Dress
#  8: Belt         9: Left-shoe   10: Right-shoe  11: Face
# 12: Left-leg    13: Right-leg   14: Left-arm    15: Right-arm
# 16: Bag         17: Scarf

# Mapping from warping body-part groups/names (classical_warping.py)
# to the ATR label IDs that should be erased when that part is warped.
WARP_PART_TO_ATR_LABELS: dict[str, list[int]] = {
    # Groups (matching BODY_PART_GROUPS in classical_warping.py)
    "arms":  [14, 15],
    "hands": [14, 15],
    "legs":  [5, 6, 12, 13],
    "feet":  [9, 10],

    # Individual parts (matching ALL_BODY_PARTS in classical_warping.py)
    "head":             [1, 2, 3, 11],
    "neck":             [17],       # Removed 11 (Face) — neck template is narrow,
                                    # erasing face would leave a hole.
    "torso":            [7, 8],     # Removed 4 (Upper-clothes) — covers the entire
                                    # shirt including sleeves, erasing it removes
                                    # the arm area. Removed 17 (Scarf) — neck area.
                                    # Warped torso template composites on top anyway.
    "right_upper_arm":  [15],
    "right_forearm":    [15],
    "left_upper_arm":   [14],
    "left_forearm":     [14],
    "right_thigh":      [6, 13],
    "right_calf":       [6, 13],
    "left_thigh":       [6, 12],
    "left_calf":        [6, 12],
    "right_foot":       [10],
    "left_foot":        [9],
    "right_hand":       [15],
    "left_hand":        [14],
}


class BodyPartSegmenter:
    """
    Human body-part semantic segmentation using SegFormer
    fine-tuned on the ATR dataset (mattmdjaga/segformer_b2_clothes).
    """

    MODEL_NAME = "mattmdjaga/segformer_b2_clothes"
    MAX_DIM = 1024  # cap input resolution to prevent OOM

    def __init__(self):
        logger.info(f"Loading body-part segmentation model: {self.MODEL_NAME}")
        self.processor = SegformerImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_NAME)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Body-part segmenter ready on {self.device}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def segment(self, image: Image.Image) -> np.ndarray:
        """
        Run human parsing on the original photo.

        Args:
            image: PIL RGB image (the original uploaded photo, NOT the silhouette).

        Returns:
            np.ndarray of shape (H, W) with integer labels 0-17 at original
            image resolution.
        """
        orig_w, orig_h = image.size

        # Cap resolution to prevent OOM on large images
        inference_image = image
        if max(orig_w, orig_h) > self.MAX_DIM:
            scale = self.MAX_DIM / max(orig_w, orig_h)
            inference_image = image.resize(
                (int(orig_w * scale), int(orig_h * scale)),
                Image.Resampling.LANCZOS,
            )

        inputs = self.processor(images=inference_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # (1, num_labels, h/4, w/4)
        # Upsample to original image size
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        labels = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        return labels.astype(np.uint8)

    # ------------------------------------------------------------------
    # Mask building
    # ------------------------------------------------------------------

    @staticmethod
    def build_erasure_mask(
        label_map: np.ndarray,
        parts_to_erase: List[str],
        dilate_px: int = 5,
        feather_px: int = 3,
    ) -> np.ndarray:
        """
        Build a soft erasure mask from a segmentation label map.

        Args:
            label_map:      (H, W) integer array of ATR labels.
            parts_to_erase: Warping part names/groups to erase
                            (e.g. ["right_foot", "left_foot"]).
            dilate_px:      Morphological dilation radius (pixels).
            feather_px:     Gaussian blur sigma for soft edges.

        Returns:
            (H, W) float32 array in [0, 1].  1.0 = fully erase this pixel.
        """
        erase_labels: set[int] = set()
        for part in parts_to_erase:
            if part in WARP_PART_TO_ATR_LABELS:
                erase_labels.update(WARP_PART_TO_ATR_LABELS[part])
            else:
                logger.warning(f"Unknown part for erasure mapping: {part}")

        if not erase_labels:
            return np.zeros(label_map.shape, dtype=np.float32)

        # Collect ATR labels for body parts NOT being erased, so we can
        # protect them from dilation bleed (e.g. arm mask dilating into neck).
        protect_labels: set[int] = set()
        parts_to_erase_set = set(parts_to_erase)
        for part_name, labels in WARP_PART_TO_ATR_LABELS.items():
            if part_name not in parts_to_erase_set:
                for lbl in labels:
                    if lbl not in erase_labels:
                        protect_labels.add(lbl)

        mask = np.isin(label_map, list(erase_labels)).astype(np.uint8)

        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        # After dilation, prevent erasure from bleeding into protected parts
        if protect_labels:
            protect_mask = np.isin(label_map, list(protect_labels))
            mask[protect_mask] = 0

        mask_float = mask.astype(np.float32)
        if feather_px > 0:
            ksize = feather_px * 2 + 1
            mask_float = cv2.GaussianBlur(mask_float, (ksize, ksize), 0)

        return np.clip(mask_float, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Silhouette erasure
    # ------------------------------------------------------------------

    @staticmethod
    def erase_parts_from_silhouette(
        silhouette: Image.Image,
        erasure_mask: np.ndarray,
    ) -> Image.Image:
        """
        Erase body parts from a silhouette by multiplying its alpha channel
        with (1 - erasure_mask).

        Args:
            silhouette:    RGBA PIL Image (the black silhouette).
            erasure_mask:  (H, W) float32 in [0, 1], 1.0 = fully erase.

        Returns:
            New RGBA PIL Image with the erased regions made transparent.
        """
        sil_array = np.array(silhouette.convert("RGBA"))

        sil_h, sil_w = sil_array.shape[:2]
        mask_h, mask_w = erasure_mask.shape[:2]
        if (mask_h, mask_w) != (sil_h, sil_w):
            erasure_mask = cv2.resize(
                erasure_mask, (sil_w, sil_h), interpolation=cv2.INTER_LINEAR
            )

        keep_factor = 1.0 - erasure_mask
        sil_array[:, :, 3] = (
            sil_array[:, :, 3].astype(np.float32) * keep_factor
        ).astype(np.uint8)

        return Image.fromarray(sil_array)
