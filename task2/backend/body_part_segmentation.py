"""
Body-Part Semantic Segmentation

Uses a SegFormer model fine-tuned on the ATR dataset to produce per-pixel
body-part labels.  The resulting label map is used to build erasure masks
that remove specific body regions from the original silhouette before
warped template parts are composited on top.
"""

import logging
from typing import List, Optional

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
    
    # Standalone classes added purely to act as "protected" labels against bleeding
    "upper_clothes":    [4],
    "face":             [11],
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
        dilate_px: int = 18,
        feather_px: int = 5,
        pose_keypoints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build a soft erasure mask from a segmentation label map.

        Args:
            label_map:      (H, W) integer array of ATR labels.
            parts_to_erase: Warping part names/groups to erase
                            (e.g. ["right_foot", "left_foot"]).
            dilate_px:      Morphological dilation radius (pixels).
            feather_px:     Gaussian blur sigma for soft edges.
            pose_keypoints: Optional (N, 3) array of OpenPose 25-point keypoints
                            (x, y, confidence). When provided and arms are being
                            erased, the hand region (below the wrists) is
                            geometrically carved out of the arm erasure mask.

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

        # Determine whether we are erasing arms (but not hands explicitly)
        arm_labels = set(WARP_PART_TO_ATR_LABELS.get("arms", []))
        hand_labels = set(WARP_PART_TO_ATR_LABELS.get("hands", []))
        erasing_arms_not_hands = (
            bool(erase_labels.intersection(arm_labels))
            and "hands" not in parts_to_erase_set
            and "right_hand" not in parts_to_erase_set
            and "left_hand" not in parts_to_erase_set
        )

        final_mask_float = np.zeros(label_map.shape[:2], dtype=np.float32)

        feet_labels = set(WARP_PART_TO_ATR_LABELS.get("feet", []))
        erasing_feet = bool(erase_labels.intersection(feet_labels))

        if erasing_feet:
            # Separate out the feet mask from other parts
            mask_other = np.isin(label_map, list(erase_labels - feet_labels)).astype(np.uint8)
            mask_feet = np.isin(label_map, list(erase_labels.intersection(feet_labels))).astype(np.uint8)

            # Standard dilation for non-feet parts
            if dilate_px > 0 and mask_other.any():
                kernel_other = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
                )
                mask_other = cv2.dilate(mask_other, kernel_other, iterations=1)

            # Two-pass dilation for feet — avoids crescent/mustache artifacts:
            # Pass 1: wider ellipse (left/right expansion)
            # Pass 2: pure downward push via a rectangle kernel anchored at its TOP row,
            #         so it ONLY pushes the mask downward, never upward.
            side_extra = 24
            down_extra = 44

            k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           ((dilate_px + side_extra) * 2 + 1,
                                            dilate_px * 2 + 1))
            mask_feet_pass1 = cv2.dilate(mask_feet, k1, iterations=1)

            k2_w = (dilate_px + side_extra) * 2 + 1
            k2_h = down_extra * 2 + 1
            k2 = np.ones((k2_h, k2_w), dtype=np.uint8)
            # anchor=(col, row): anchoring at row=0 (top of kernel) means the
            # kernel extends BELOW each source pixel only.
            mask_feet_pass2 = cv2.dilate(mask_feet, k2,
                                         anchor=(k2_w // 2, 0),
                                         iterations=1)

            mask_feet_dilated = np.maximum(mask_feet_pass1, mask_feet_pass2)

            # Very small feather for feet (crisp bottom edge); normal for others
            mask_other_float = mask_other.astype(np.float32)
            mask_feet_float = mask_feet_dilated.astype(np.float32)

            if feather_px > 0:
                ksize_other = feather_px * 2 + 1
                mask_other_float = cv2.GaussianBlur(mask_other_float, (ksize_other, ksize_other), 0)
                mask_feet_float = cv2.GaussianBlur(mask_feet_float, (3, 3), 0)

            final_mask_float = np.maximum(mask_other_float, mask_feet_float)

        else:
            mask = np.isin(label_map, list(erase_labels)).astype(np.uint8)
            if dilate_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
                )
                mask = cv2.dilate(mask, kernel, iterations=1)

            mask_float = mask.astype(np.float32)
            if feather_px > 0:
                ksize = feather_px * 2 + 1
                mask_float = cv2.GaussianBlur(mask_float, (ksize, ksize), 0)
            final_mask_float = mask_float

        # ---- Hand geometric exclusion ----------------------------------------
        # ATR labels 14/15 cover the full arm+hand.  When we're erasing arms but
        # NOT hands, carve out an approximate hand region around each wrist using
        # the OpenPose wrist and elbow keypoints.
        # OpenPose indices: 3=r_elbow, 4=r_wrist, 6=l_elbow, 7=l_wrist
        if erasing_arms_not_hands and pose_keypoints is not None and len(pose_keypoints) > 0:
            H, W = label_map.shape[:2]
            hand_carve = np.zeros((H, W), dtype=np.uint8)

            # Hand radius covers full fist area (approx 10% of image height)
            hand_radius = max(30, int(H * 0.10))

            for elbow_idx, wrist_idx in ((3, 4), (6, 7)):  # (r_elbow, r_wrist), (l_elbow, l_wrist)
                if wrist_idx < len(pose_keypoints):
                    wx, wy, wconf = pose_keypoints[wrist_idx]
                    if wconf > 0.05:
                        # Default hand center: at the wrist
                        hand_cx, hand_cy = float(wx), float(wy)

                        # If elbow is visible, offset the circle CENTER past the wrist
                        # toward the fingertips (hand extends ~1 hand-radius past wrist)
                        if elbow_idx < len(pose_keypoints):
                            ex, ey, econf = pose_keypoints[elbow_idx]
                            if econf > 0.05:
                                # Direction vector from elbow → wrist (normalized)
                                dx = float(wx) - float(ex)
                                dy = float(wy) - float(ey)
                                length = max((dx**2 + dy**2) ** 0.5, 1.0)
                                dx /= length
                                dy /= length
                                # Shift center by hand_radius * 0.8 past the wrist
                                hand_cx = float(wx) + dx * hand_radius * 0.8
                                hand_cy = float(wy) + dy * hand_radius * 0.8

                        cx, cy = int(hand_cx), int(hand_cy)
                        cv2.circle(hand_carve, (cx, cy), hand_radius, 1, -1)

            if hand_carve.any():
                # NO erosion — we want the full carve region preserved
                final_mask_float[hand_carve == 1] = 0.0
                logger.debug(f"Carved out hand region: {hand_carve.sum()} pixels protected")

        # After dilation and feathering, prevent erasure from bleeding into protected parts
        if protect_labels:
            protect_mask = np.isin(label_map, list(protect_labels))
            final_mask_float[protect_mask] = 0.0

        return np.clip(final_mask_float, 0.0, 1.0)

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
