"""
Classical Warping Engine

Warps body part templates onto detected pose keypoints to create stylized silhouettes.
Refactored from generate_silhouettes.py for clean API integration.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from PIL import Image
import logging

logger = logging.getLogger("ClassicalWarping")

# Body part group mappings for convenience
BODY_PART_GROUPS = {
    "arms": ["right_upper_arm", "right_forearm", "left_upper_arm", "left_forearm"],
    "legs": ["right_thigh", "right_calf", "left_thigh", "left_calf"],
    "feet": ["right_foot", "left_foot"],
    "hands": ["right_hand", "left_hand"],
}

# All available body parts
ALL_BODY_PARTS = [
    "torso", "neck", "head",
    "right_upper_arm", "right_forearm", "left_upper_arm", "left_forearm",
    "right_thigh", "right_calf", "left_thigh", "left_calf",
    "right_foot", "left_foot", "right_hand", "left_hand",
]


def expand_parts_list(parts: list[str]) -> list[str]:
    """Expand group names (like 'arms') into individual body parts."""
    expanded = []
    for part in parts:
        if part in BODY_PART_GROUPS:
            expanded.extend(BODY_PART_GROUPS[part])
        elif part in ALL_BODY_PARTS:
            expanded.append(part)
        else:
            logger.warning(f"Unknown body part: {part}")
    return list(set(expanded))  # Remove duplicates


# Keypoint format mapping
# DWPose outputs 18 keypoints in OpenPose format (NOT COCO):
# 0: nose, 1: neck, 2: r_shoulder, 3: r_elbow, 4: r_wrist,
# 5: l_shoulder, 6: l_elbow, 7: l_wrist, 8: mid_hip,
# 9: r_hip, 10: r_knee, 11: r_ankle, 12: l_hip, 13: l_knee, 14: l_ankle,
# 15: r_eye, 16: l_eye, 17: (duplicate neck in DWPose output)
#
# OpenPose 25-point format extends this with:
# 17: r_ear, 18: l_ear, 19-21: left foot (toe, small_toe, heel),
# 22-24: right foot (toe, small_toe, heel)
#
# Since DWPose outputs OpenPose 18-point format, most indices match.
# Only indices 17, 18 (ears) and 19-24 (feet details) need fallback.

# For 18-point DWPose format, map missing indices to available ones
OPENPOSE_18_FALLBACK = {
    17: -1,  # r_ear: Not available in 18-point, will use head
    18: -1,  # l_ear: Not available in 18-point, will use head
    19: 14,  # left_big_toe -> left_ankle (fallback)
    20: 14,  # left_small_toe -> left_ankle
    21: 14,  # left_heel -> left_ankle
    22: 11,  # right_big_toe -> right_ankle
    23: 11,  # right_small_toe -> right_ankle
    24: 11,  # right_heel -> right_ankle
}


class ClassicalWarpingEngine:
    """
    Engine for warping pre-defined body part templates onto detected poses.
    """

    def __init__(self, links_dir: Path):
        """
        Initialize with a directory containing body part templates.
        
        Args:
            links_dir: Path to directory with links.csv and PNG images
        """
        self.links_dir = Path(links_dir)
        self.links_dict = self._parse_links_dir()
        logger.info(f"Loaded {len(self.links_dict)} body part templates from {links_dir}")

    def _parse_links_dir(self) -> dict:
        """Load template images and reference keypoints from CSV."""
        links_csv_path = self.links_dir / "links.csv"
        if not links_csv_path.exists():
            raise FileNotFoundError(f"links.csv not found in {self.links_dir}")

        links_df = pd.read_csv(links_csv_path)
        links_dict = {}

        for _, row in links_df.iterrows():
            link_name = row["name"]
            link_data = {
                "pt1_idx": row["pt1_idx"],
                "pt2_idx": row["pt2_idx"],
                "pt1": np.array([row["pt1_x"], row["pt1_y"]]),
                "pt2": np.array([row["pt2_x"], row["pt2_y"]]),
            }

            link_img_path = self.links_dir / f"{link_name}.png"
            if link_img_path.exists():
                link_data["image"] = cv2.imread(str(link_img_path), cv2.IMREAD_UNCHANGED)
                link_data["length"] = np.linalg.norm(link_data["pt2"] - link_data["pt1"])
                links_dict[link_name] = link_data
            else:
                logger.warning(f"Template image not found: {link_img_path}")

        # Calculate relative lengths (normalized to torso)
        if "torso" in links_dict:
            torso_length = links_dict["torso"]["length"]
            for link_name, link_data in links_dict.items():
                links_dict[link_name]["relative_length"] = link_data["length"] / torso_length

        return links_dict

    def _get_hand_middle_keypoint(self, hand_keypoints: np.ndarray) -> np.ndarray:
        """Calculate middle point of hand from keypoints."""
        if len(hand_keypoints) < 21:
             return np.array([0, 0])
        idxes = [1, 5, 9, 13, 17]
        visible_keypoints = [hand_keypoints[i] for i in idxes if hand_keypoints[i][2] != 0]
        if not visible_keypoints:
            return hand_keypoints[0][:2]  # Fall back to wrist
        visible_keypoints = np.array(visible_keypoints)
        return np.mean(visible_keypoints[:, :2], axis=0)

    def _get_head_middle_keypoint(self, keypoints: np.ndarray) -> np.ndarray:
        """Calculate head middle from ear keypoints."""
        r_ear = keypoints[17] if len(keypoints) > 17 else np.array([0, 0, 0])
        l_ear = keypoints[18] if len(keypoints) > 18 else np.array([0, 0, 0])

        if r_ear[2] != 0 and l_ear[2] != 0:
            return (r_ear[:2] + l_ear[:2]) / 2
        elif r_ear[2] != 0:
            return r_ear[:2]
        elif l_ear[2] != 0:
            return l_ear[:2]
        else:
            # Fall back to nose if available
            nose = keypoints[0] if len(keypoints) > 0 else np.array([0, 0, 0])
            return nose[:2]

    def _get_head_middle_keypoint_mapped(self, get_keypoint_fn) -> np.ndarray:
        """Calculate head middle using the index-mapped get_keypoint function.
        
        Uses ears first (like original code) as they give a horizontal reference
        matching the head template orientation. Falls back to eyes/nose if ears unavailable.
        """
        # OpenPose 25: 17 = r_ear, 18 = l_ear, 15 = r_eye, 16 = l_eye, 0 = nose
        r_ear = get_keypoint_fn(17)
        l_ear = get_keypoint_fn(18)
        r_eye = get_keypoint_fn(15)
        l_eye = get_keypoint_fn(16)
        nose = get_keypoint_fn(0)

        # Prefer ears (horizontal reference matching template)
        if r_ear[2] > 0.3 and l_ear[2] > 0.3:
            return (r_ear[:2] + l_ear[:2]) / 2
        elif r_ear[2] > 0.3:
            return r_ear[:2]
        elif l_ear[2] > 0.3:
            return l_ear[:2]
        # Fall back to eyes
        elif r_eye[2] > 0.3 and l_eye[2] > 0.3:
            return (r_eye[:2] + l_eye[:2]) / 2
        elif r_eye[2] > 0.3:
            return r_eye[:2]
        elif l_eye[2] > 0.3:
            return l_eye[:2]
        # Fall back to nose
        elif nose[2] > 0.3:
            return nose[:2]
        else:
            # Last resort: neck
            neck = get_keypoint_fn(1)
            return neck[:2]

    def _parse_keypoints(self, person_dict: dict) -> dict:
        """
        Extract destination keypoints from person dict.
        Expects OpenPose 25-point format (from pose_estimator conversion).
        
        Args:
            person_dict: Dict with pose_keypoints_2d, hand_left_keypoints_2d, 
                        hand_right_keypoints_2d
        """
        # Parse body keypoints - now expecting OpenPose 25 format
        pose_kp = person_dict.get("pose_keypoints_2d", [])
        keypoints = np.array(pose_kp).reshape((-1, 3)) if pose_kp else np.zeros((25, 3))

        num_keypoints = len(keypoints)
        logger.debug(f"Number of body keypoints: {num_keypoints}")

        # Extract wrists/hands for hand placement and orientation
        left_hand_kp = person_dict.get("hand_left_keypoints_2d", [])
        right_hand_kp = person_dict.get("hand_right_keypoints_2d", [])
        
        left_hand = np.array(left_hand_kp).reshape((-1, 3)) if left_hand_kp else np.zeros((0, 3))
        right_hand = np.array(right_hand_kp).reshape((-1, 3)) if right_hand_kp else np.zeros((0, 3))

        # Get keypoint accessor function
        def get_keypoint(idx):
            """
            Get keypoint by OpenPose index.
            DWPose outputs OpenPose 18-point format directly.
            Indices 0-16 are available, 17+ need fallback.
            """
            if idx < 0:
                return np.array([0, 0, 0])
            
            # Direct access for available keypoints (0-16)
            if idx < num_keypoints:
                return keypoints[idx]
            
            # Fallback for missing keypoints (17+)
            fallback_idx = OPENPOSE_18_FALLBACK.get(idx, -1)
            
            # Special handling for mid_hip (index 8) if not available
            if idx == 8 and (idx >= num_keypoints or keypoints[idx][2] == 0):
                # Average of left and right hips
                l_hip = keypoints[12] if 12 < num_keypoints else np.array([0, 0, 0])
                r_hip = keypoints[9] if 9 < num_keypoints else np.array([0, 0, 0])
                if l_hip[2] > 0 and r_hip[2] > 0:
                    return np.array([(l_hip[0] + r_hip[0]) / 2,
                                    (l_hip[1] + r_hip[1]) / 2, 1.0])
                elif l_hip[2] > 0:
                    return l_hip
                elif r_hip[2] > 0:
                    return r_hip
            
            if fallback_idx >= 0 and fallback_idx < num_keypoints:
                return keypoints[fallback_idx]
            
            return np.array([0, 0, 0])

        keypoints_dict = {}
        head_middle = None

        for link_name, link_data in self.links_dict.items():
            # Determine destination points based on link type
            if link_name == "neck":
                dst_pt1 = get_keypoint(link_data["pt1_idx"])[:2]
                if head_middle is None:
                    head_middle = self._get_head_middle_keypoint_mapped(get_keypoint)
                dst_pt2 = head_middle
            elif link_name == "head":
                if head_middle is None:
                    head_middle = self._get_head_middle_keypoint_mapped(get_keypoint)
                # Use neck→head_middle for orientation (stable regardless of face direction)
                # instead of head_middle→nose (varies with where person looks)
                neck_pt = get_keypoint(1)[:2]
                dst_pt1 = neck_pt
                dst_pt2 = head_middle
            elif link_name == "right_hand":
                dst_pt1 = right_hand[0][:2] if len(right_hand) > 0 else np.array([0, 0])
                dst_pt2 = self._get_hand_middle_keypoint(right_hand)
            elif link_name == "left_hand":
                dst_pt1 = left_hand[0][:2] if len(left_hand) > 0 else np.array([0, 0])
                dst_pt2 = self._get_hand_middle_keypoint(left_hand)
            else:
                pt1_idx = link_data["pt1_idx"]
                pt2_idx = link_data["pt2_idx"]
                dst_pt1 = get_keypoint(pt1_idx)[:2]
                dst_pt2 = get_keypoint(pt2_idx)[:2]

            length = np.linalg.norm(dst_pt2 - dst_pt1)
            keypoints_dict[link_name] = {
                "length": length,
                "dst_pt1": dst_pt1,
                "dst_pt2": dst_pt2,
            }

        # Calculate relative lengths
        if "torso" in keypoints_dict and keypoints_dict["torso"]["length"] > 0:
            torso_length = keypoints_dict["torso"]["length"]
            for link_name, kp_data in keypoints_dict.items():
                kp_data["relative_length"] = kp_data["length"] / torso_length

            # Adjust head length to match template proportions
            if "head" in keypoints_dict and "head" in self.links_dict:
                head_length = keypoints_dict["head"]["length"]
                if keypoints_dict["head"]["relative_length"] > 0:
                    desired_head_length = (
                        self.links_dict["head"]["relative_length"]
                        / keypoints_dict["head"]["relative_length"]
                        * head_length
                    )
                    dst_pt1 = keypoints_dict["head"]["dst_pt1"]
                    dst_pt2 = keypoints_dict["head"]["dst_pt2"]
                    head_vec = dst_pt2 - dst_pt1
                    if np.linalg.norm(head_vec) > 0:
                        head_vec_normalized = head_vec / np.linalg.norm(head_vec)
                        keypoints_dict["head"]["dst_pt2"] = dst_pt1 + head_vec_normalized * desired_head_length
                        keypoints_dict["head"]["length"] = desired_head_length

        return keypoints_dict

    def _get_transformation_matrix(
        self, src_pt1: np.ndarray, src_pt2: np.ndarray,
        dst_pt1: np.ndarray, dst_pt2: np.ndarray,
        width_scale_coeff: float = 1.0
    ) -> np.ndarray:
        """
        Compute affine transformation matrix to warp a body part.
        
        Transform: translate to origin → rotate flat → scale → rotate to target → translate to dest
        """
        src_vec = src_pt2 - src_pt1
        dst_vec = dst_pt2 - dst_pt1

        src_len = np.linalg.norm(src_vec)
        dst_len = np.linalg.norm(dst_vec)

        if src_len == 0 or dst_len == 0:
            return None

        src_angle = np.arctan2(src_vec[1], src_vec[0])
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

        # 1. Translate src_pt1 to origin
        t1 = np.array([[1, 0, -src_pt1[0]], [0, 1, -src_pt1[1]], [0, 0, 1]])

        # 2. Rotate to align with x-axis
        r1 = np.array([
            [np.cos(-src_angle), -np.sin(-src_angle), 0],
            [np.sin(-src_angle), np.cos(-src_angle), 0],
            [0, 0, 1],
        ])

        # 3. Scale (length on x, width on y)
        s_l = dst_len / src_len
        s_w = s_l * width_scale_coeff
        s = np.array([[s_l, 0, 0], [0, s_w, 0], [0, 0, 1]])

        # 4. Rotate to target angle
        r2 = np.array([
            [np.cos(dst_angle), -np.sin(dst_angle), 0],
            [np.sin(dst_angle), np.cos(dst_angle), 0],
            [0, 0, 1],
        ])

        # 5. Translate to destination
        t2 = np.array([[1, 0, dst_pt1[0]], [0, 1, dst_pt1[1]], [0, 0, 1]])

        # Compose: t2 @ r2 @ s @ r1 @ t1
        m = t2 @ r2 @ s @ r1 @ t1
        return m[:2, :]

    def _get_head_transformation(
        self, link_img: np.ndarray, link_data: dict, 
        kp_data: dict, keypoints_dict: dict, width_scale_factor: float
    ) -> Optional[np.ndarray]:
        """
        Special transformation for head - uses uniform scaling based on body proportions
        and centers the head image on head_middle with body-aligned rotation.
        Automatically flips head based on body facing direction.
        """
        h_img, w_img = link_img.shape[:2]
        
        # Determine if we need to flip the head based on body facing direction
        # Template head faces right (template center to nose is roughly horizontal right)
        # Flip if person is facing left
        flip_head = False
        if "right_upper_arm" in keypoints_dict and "left_upper_arm" in keypoints_dict:
            # Get shoulder positions from arm keypoints
            r_shoulder = keypoints_dict["right_upper_arm"]["dst_pt1"]
            l_shoulder = keypoints_dict["left_upper_arm"]["dst_pt1"]
            shoulder_center_x = (r_shoulder[0] + l_shoulder[0]) / 2
            
            # Get nose position from head keypoints (dst_pt2 was set to nose before we changed to head_middle)
            # We need to get nose separately - it's at keypoint index 0
            # For now, use a simple heuristic: if right shoulder is to the left of left shoulder, person is facing away
            if r_shoulder[0] > l_shoulder[0]:  # Normal orientation (right shoulder on right)
                # Template matches, no flip needed
                flip_head = False
            else:  # Person rotated (right shoulder on left side of image)
                flip_head = True
        
        if flip_head:
            link_img = cv2.flip(link_img, 1)  # Horizontal flip
        
        # Template center (where pt1 is for head = head_middle in template coords)
        # Use approximate center of head template based on reference points
        src_pt1 = link_data["pt1"].copy()  # template head_middle point
        src_pt2 = link_data["pt2"].copy()  # template nose point
        
        if flip_head:
            # Adjust source points for flipped image
            src_pt1[0] = w_img - src_pt1[0]
            src_pt2[0] = w_img - src_pt2[0]
        
        # Use bottom-center of head template as anchor point for neck connection
        # The bottom of the head template is where it connects to the neck
        template_neck_anchor = np.array([w_img / 2, h_img - 5])  # bottom-center with small margin
        
        # Destination: anchor to neck position (dst_pt1 is neck, dst_pt2 is head_middle)
        # We want the bottom of the head to be at the neck keypoint
        neck_pos = kp_data["dst_pt1"]  # This is neck position
        
        # Scale: based on neck→head_middle distance compared to template's relative length
        if "torso" in keypoints_dict:
            torso_len = keypoints_dict["torso"]["length"]
            head_len = kp_data["length"]  # neck→head_middle distance
            
            # Template proportions: head relative_length tells us expected size
            template_head_rel = link_data.get("relative_length", 0.3)
            template_torso_len = self.links_dict["torso"]["length"]
            expected_head_len = template_head_rel * torso_len
            
            # Scale to make template fill the expected head size
            template_head_ref_len = link_data["length"]  # pt1→pt2 distance in template
            scale = expected_head_len / template_head_ref_len if template_head_ref_len > 0 else 1.0
            # Reduce head size to 70% for better proportions (head was too large)
            scale = scale * 0.7
            # Clamp scale to reasonable range
            scale = np.clip(scale, 0.3, 3.0)
        else:
            scale = 1.0
        
        # Rotation: use torso tilt for head orientation (body-aligned)
        rotation = 0.0
        if "torso" in keypoints_dict:
            torso_pt1 = keypoints_dict["torso"]["dst_pt1"]  # neck
            torso_pt2 = keypoints_dict["torso"]["dst_pt2"]  # mid_hip
            torso_vec = torso_pt2 - torso_pt1
            # Torso angle from vertical (should be small for upright person)
            torso_angle = np.arctan2(torso_vec[0], torso_vec[1])  # deviation from vertical
            rotation = torso_angle  # Apply same tilt to head
        
        # Apply slight scaling based on width
        scale_w = scale * width_scale_factor
        scale_h = scale
        
        # Build transformation: translate neck anchor to origin → scale → rotate → translate to neck
        # 1. Translate template neck anchor to origin
        t1 = np.array([[1, 0, -template_neck_anchor[0]], 
                       [0, 1, -template_neck_anchor[1]], 
                       [0, 0, 1]])
        
        # 2. Scale
        s = np.array([[scale_w, 0, 0], 
                      [0, scale_h, 0], 
                      [0, 0, 1]])
        
        # 3. Rotate
        r = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                      [np.sin(rotation), np.cos(rotation), 0],
                      [0, 0, 1]])
        
        # 4. Translate to neck position
        t2 = np.array([[1, 0, neck_pos[0]], 
                       [0, 1, neck_pos[1]], 
                       [0, 0, 1]])
        
        m = t2 @ r @ s @ t1
        return m[:2, :]

    def generate_silhouette(
        self,
        person_dict: dict,
        canvas_size: tuple[int, int],
        parts_to_warp: Optional[list[str]] = None,
        width_scale_factor: float = 0.85,
    ) -> Image.Image:
        """
        Generate a warped silhouette from body part templates.
        
        Args:
            person_dict: OpenPose-format dict with keypoints for one person
            canvas_size: (width, height) of the output canvas
            parts_to_warp: List of body parts/groups to warp. None = all parts.
            width_scale_factor: Multiplier for limb thickness (default 0.85)
            
        Returns:
            PIL Image (RGBA) with warped body parts on transparent background
        """
        w, h = canvas_size

        # Determine which parts to warp
        if parts_to_warp is None:
            parts_to_warp = list(self.links_dict.keys())
        else:
            parts_to_warp = expand_parts_list(parts_to_warp)

        # Define Z-order priority (items not in list go at the end)
        # 1. Neck (back)
        # 2. Legs/Feet (behind/mid)
        # 3. Torso (mid)
        # 4. Arms (front)
        # 5. Head (front)
        # 6. Hands (most front)
        Z_ORDER = [
            "neck", 
            "left_thigh", "right_thigh", "left_calf", "right_calf", "left_foot", "right_foot",
            "torso", 
            "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
            "head",
            "left_hand", "right_hand"
        ]
        
        # Sort parts_to_warp based on index in Z_ORDER to ensure correct layering
        def z_sort_key(part_name):
            try:
                return Z_ORDER.index(part_name)
            except ValueError:
                return 999
        
        parts_to_warp.sort(key=z_sort_key)

        # Parse keypoints
        keypoints_dict = self._parse_keypoints(person_dict)

        # Create transparent canvas (BGRA for cv2)
        canvas = np.zeros((h, w, 4), dtype=np.uint8)

        for link_name in parts_to_warp:
            if link_name not in self.links_dict:
                continue
            if link_name not in keypoints_dict:
                continue

            link_data = self.links_dict[link_name]
            kp_data = keypoints_dict[link_name]

            # Skip if keypoints are invalid (zero length)
            if kp_data["length"] < 1:
                continue

            src_pt1 = link_data["pt1"]
            src_pt2 = link_data["pt2"]
            dst_pt1 = kp_data["dst_pt1"]
            dst_pt2 = kp_data["dst_pt2"]

            # Calculate width scale coefficient
            width_scale_coeff = width_scale_factor
            if "relative_length" in link_data and "relative_length" in kp_data:
                if kp_data["relative_length"] > 0:
                    width_scale_coeff = (
                        link_data["relative_length"]
                        / kp_data["relative_length"]
                        * width_scale_factor
                    )

            link_img = link_data["image"]
            
            # Special handling for head - use scaling based on torso, centered on head_middle
            if link_name == "head":
                m = self._get_head_transformation(
                    link_img, link_data, kp_data, keypoints_dict, width_scale_factor
                )
            else:
                # Check if we need to flip the template based on limb direction
                src_vec = src_pt2 - src_pt1
                dst_vec = dst_pt2 - dst_pt1
                
                # Special handling for feet: flatten them (make them more horizontal)
                if link_name in ["right_foot", "left_foot"]:
                    # Get current angle
                    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
                    dst_len = np.linalg.norm(dst_vec)
                    
                    # Clamp angle to be horizontal-ish (within 20 degrees / 0.35 rad)
                    # We want to preserve left/right direction (0 or pi)
                    if abs(dst_angle) < np.pi/2: # Pointing Right
                        # Clamp to [-0.35, 0.35]
                        new_angle = np.clip(dst_angle, -0.35, 0.35)
                    else: # Pointing Left
                        # Angle is large, e.g. 3, -3. Normalize to [-pi, pi]
                        # We want it close to pi or -pi
                        if dst_angle > 0:
                            new_angle = np.clip(dst_angle, np.pi - 0.35, np.pi)
                        else:
                            new_angle = np.clip(dst_angle, -np.pi, -np.pi + 0.35)
                    
                    # Recompute dst_pt2 based on flattened angle
                    new_vec = np.array([np.cos(new_angle), np.sin(new_angle)]) * dst_len
                    dst_pt2 = dst_pt1 + new_vec
                    dst_vec = new_vec # Update vector for flip check below
                
                src_angle = np.arctan2(src_vec[1], src_vec[0])
                dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
                
                # Angle difference - if > 90°, template points in significantly different direction
                angle_diff = abs(dst_angle - src_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                # Flip template horizontally if direction differs by more than 90°
                if angle_diff > np.pi / 2:
                    link_img = cv2.flip(link_img, 1)  # Horizontal flip
                    # Adjust source points for flipped image
                    img_w = link_img.shape[1]
                    src_pt1_flipped = np.array([img_w - src_pt1[0], src_pt1[1]])
                    src_pt2_flipped = np.array([img_w - src_pt2[0], src_pt2[1]])
                    src_pt1, src_pt2 = src_pt1_flipped, src_pt2_flipped
                
                m = self._get_transformation_matrix(
                    src_pt1, src_pt2, dst_pt1, dst_pt2, width_scale_coeff
                )
            
            if m is None:
                continue

            warped_link = cv2.warpAffine(
                link_img, m, (w, h), flags=cv2.INTER_LINEAR,
            )

            # Composite with alpha blending
            if warped_link.shape[2] == 4:
                link_alpha = warped_link[:, :, 3:4] / 255.0
                link_bgr = warped_link[:, :, :3]

                # Blend onto canvas
                canvas_bgr = canvas[:, :, :3]
                canvas_alpha = canvas[:, :, 3:4] / 255.0

                # Over compositing
                out_alpha = link_alpha + canvas_alpha * (1 - link_alpha)
                out_alpha_safe = np.where(out_alpha == 0, 1, out_alpha)
                out_bgr = (link_bgr * link_alpha + canvas_bgr * canvas_alpha * (1 - link_alpha)) / out_alpha_safe

                canvas[:, :, :3] = out_bgr.astype(np.uint8)
                canvas[:, :, 3] = (out_alpha * 255).astype(np.uint8).squeeze()

        # Convert BGRA to RGBA for PIL
        canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(canvas_rgba)


# Module-level instance for convenience
_default_engine: Optional[ClassicalWarpingEngine] = None


def get_default_engine() -> ClassicalWarpingEngine:
    """Get or create the default warping engine with Prince_Achmed templates."""
    global _default_engine
    if _default_engine is None:
        default_links_dir = Path(__file__).parent / "Prince_Achmed" / "links"
        _default_engine = ClassicalWarpingEngine(default_links_dir)
    return _default_engine
