"""
Classical Warping Engine

Warps body part templates onto detected pose keypoints to create stylized silhouettes.
Mirrors generate_silhouettes.py logic, adapted for the FastAPI backend.
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


# OpenPose 25-point keypoint indices:
# 0: nose,  1: neck,        2: r_shoulder, 3: r_elbow,  4: r_wrist,
# 5: l_shoulder, 6: l_elbow, 7: l_wrist,   8: mid_hip,
# 9: r_hip, 10: r_knee,     11: r_ankle,   12: l_hip,  13: l_knee, 14: l_ankle,
# 15: r_eye, 16: l_eye,     17: r_ear,     18: l_ear,
# 19: l_big_toe, 20: l_small_toe, 21: l_heel,
# 22: r_big_toe, 23: r_small_toe, 24: r_heel

# Fallback for extended foot keypoints that may be absent
OPENPOSE_25_FALLBACK = {
    19: 14,  # left_big_toe   -> left_ankle
    20: 14,  # left_small_toe -> left_ankle
    21: 14,  # left_heel      -> left_ankle
    22: 11,  # right_big_toe  -> right_ankle
    23: 11,  # right_small_toe-> right_ankle
    24: 11,  # right_heel     -> right_ankle
}


class ClassicalWarpingEngine:
    """
    Engine for warping pre-defined body part templates onto detected poses.
    """

    def __init__(self, links_dir: Path):
        """
        Initialize with a directory containing body part templates.

        Args:
            links_dir: Path to directory with links.csv and PNG images.
        """
        self.links_dir = Path(links_dir)
        self.links_dict = self._parse_links_dir()
        logger.info(f"Loaded {len(self.links_dict)} body part templates from {links_dir}")

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _parse_links_dir(self) -> dict:
        """Load template images, reference keypoints, and direction from CSV."""
        links_csv_path = self.links_dir / "links.csv"
        if not links_csv_path.exists():
            raise FileNotFoundError(f"links.csv not found in {self.links_dir}")

        links_df = pd.read_csv(links_csv_path)
        links_dict = {}

        for _, row in links_df.iterrows():
            link_name = row["name"]
            link_data = {
                "pt1_idx":  row["pt1_idx"],
                "pt2_idx":  row["pt2_idx"],
                "pt1":      np.array([row["pt1_x"], row["pt1_y"]]),
                "pt2":      np.array([row["pt2_x"], row["pt2_y"]]),
                "direction": row["direction"],  # 'l' or 'r'
            }

            link_img_path = self.links_dir / f"{link_name}.png"
            if link_img_path.exists():
                link_data["image"] = cv2.imread(str(link_img_path), cv2.IMREAD_UNCHANGED)
                link_data["length"] = np.linalg.norm(link_data["pt2"] - link_data["pt1"])
                links_dict[link_name] = link_data
            else:
                logger.warning(f"Template image not found: {link_img_path}")

        # Relative lengths (normalised to torso)
        if "torso" in links_dict:
            torso_length = links_dict["torso"]["length"]
            for ld in links_dict.values():
                ld["relative_length"] = ld["length"] / torso_length

        return links_dict

    # ------------------------------------------------------------------
    # Helper utilities (matching generate_silhouettes.py)
    # ------------------------------------------------------------------

    def _flip_link(self, link_data: dict) -> dict:
        """Return a copy of link_data with image flipped horizontally and pts adjusted."""
        flipped = link_data.copy()
        flipped_image = cv2.flip(link_data["image"], 1)
        flipped["image"] = flipped_image
        img_w = link_data["image"].shape[1]
        flipped["pt1"] = np.array([img_w - link_data["pt1"][0], link_data["pt1"][1]])
        flipped["pt2"] = np.array([img_w - link_data["pt2"][0], link_data["pt2"][1]])
        return flipped

    def _get_head_direction(self, get_keypoint_fn) -> str:
        """
        Determine which direction the head is facing ('l' or 'r').

        Uses ear/nose projection, mirroring get_head_direction() from
        generate_silhouettes.py.  If the nose projects past the midpoint of the
        ear-to-ear vector it faces right, otherwise left.
        """
        r_ear = get_keypoint_fn(17)
        l_ear = get_keypoint_fn(18)
        nose  = get_keypoint_fn(0)[:2]

        if r_ear[2] > 0 and l_ear[2] > 0:
            ear_vec = l_ear[:2] - r_ear[:2]
            denom = np.dot(ear_vec, ear_vec)
            if denom > 0:
                proj = np.dot(nose - r_ear[:2], ear_vec) / denom
                return "l" if proj < 0.5 else "r"
        # Single ear visible
        if r_ear[2] > 0:
            return "r"
        return "l"

    def _get_hand_middle_keypoint(self, hand_keypoints: np.ndarray) -> np.ndarray:
        """Calculate the middle point of a hand from its keypoints."""
        if len(hand_keypoints) < 21:
            return hand_keypoints[0][:2] if len(hand_keypoints) > 0 else np.array([0.0, 0.0])
        idxes = [1, 5, 9, 13, 17]
        visible = [hand_keypoints[i] for i in idxes if hand_keypoints[i][2] != 0]
        if not visible:
            return hand_keypoints[0][:2]
        return np.mean(np.array(visible)[:, :2], axis=0)

    def _get_head_middle_keypoint_mapped(self, get_keypoint_fn) -> np.ndarray:
        """
        Return the midpoint between the ears (head centre).
        Falls back to eyes then nose/neck when ears are not visible.
        """
        r_ear = get_keypoint_fn(17)
        l_ear = get_keypoint_fn(18)
        r_eye = get_keypoint_fn(15)
        l_eye = get_keypoint_fn(16)
        nose  = get_keypoint_fn(0)

        if r_ear[2] > 0.3 and l_ear[2] > 0.3:
            return (r_ear[:2] + l_ear[:2]) / 2
        elif r_ear[2] > 0.3:
            return r_ear[:2]
        elif l_ear[2] > 0.3:
            return l_ear[:2]
        elif r_eye[2] > 0.3 and l_eye[2] > 0.3:
            return (r_eye[:2] + l_eye[:2]) / 2
        elif r_eye[2] > 0.3:
            return r_eye[:2]
        elif l_eye[2] > 0.3:
            return l_eye[:2]
        elif nose[2] > 0.3:
            return nose[:2]
        else:
            return get_keypoint_fn(1)[:2]  # neck as last resort

    def _adjust_link_direction(self, links_dict: dict, keypoints_dict: dict) -> dict:
        """
        Flip arm / leg / torso templates so their rendered direction matches the pose.
        Mirrors adjust_link_direction() from generate_silhouettes.py.
        Operates on *links_dict* (the per-call working copy) in-place and returns it.
        """
        arm_keys = ["right_upper_arm", "right_forearm", "left_upper_arm", "left_forearm"]
        if all(k in keypoints_dict for k in arm_keys):
            # Vectors as defined in the original script
            r_upper_v = (np.array(keypoints_dict["right_upper_arm"]["dst_pt1"])
                       - np.array(keypoints_dict["right_upper_arm"]["dst_pt2"]))
            r_fore_v  = (np.array(keypoints_dict["right_forearm"]["dst_pt2"])
                       - np.array(keypoints_dict["right_forearm"]["dst_pt1"]))
            l_upper_v = (np.array(keypoints_dict["left_upper_arm"]["dst_pt1"])
                       - np.array(keypoints_dict["left_upper_arm"]["dst_pt2"]))
            l_fore_v  = (np.array(keypoints_dict["left_forearm"]["dst_pt2"])
                       - np.array(keypoints_dict["left_forearm"]["dst_pt1"]))

            r_upper_angle = np.degrees(np.arctan2(r_upper_v[0], r_upper_v[1]))
            r_fore_angle  = np.degrees(np.arctan2(r_fore_v[0],  r_fore_v[1]))
            l_upper_angle = np.degrees(np.arctan2(l_upper_v[0], l_upper_v[1]))
            l_fore_angle  = np.degrees(np.arctan2(l_fore_v[0],  l_fore_v[1]))

            r_elbow_angle = (r_upper_angle - r_fore_angle) % 360
            l_elbow_angle = (l_upper_angle - l_fore_angle) % 360

            if "right_forearm" in links_dict:
                if ((r_elbow_angle > 180 and links_dict["right_forearm"]["direction"] == "r") or
                        (r_elbow_angle < 180 and links_dict["right_forearm"]["direction"] == "l")):
                    for part in ["right_upper_arm", "right_forearm", "right_hand"]:
                        if part in links_dict:
                            links_dict[part] = self._flip_link(links_dict[part])

            if "left_forearm" in links_dict:
                if ((l_elbow_angle > 180 and links_dict["left_forearm"]["direction"] == "r") or
                        (l_elbow_angle < 180 and links_dict["left_forearm"]["direction"] == "l")):
                    for part in ["left_upper_arm", "left_forearm", "left_hand"]:
                        if part in links_dict:
                            links_dict[part] = self._flip_link(links_dict[part])

            # Torso
            if "torso" in links_dict:
                if (((r_elbow_angle > 180 and l_elbow_angle > 180)
                        and links_dict["torso"]["direction"] == "r") or
                        ((r_elbow_angle < 180 and l_elbow_angle < 180)
                        and links_dict["torso"]["direction"] == "l")):
                    links_dict["torso"] = self._flip_link(links_dict["torso"])

        # Legs
        if all(k in keypoints_dict for k in ["left_foot", "right_foot"]):
            l_foot_v = (np.array(keypoints_dict["left_foot"]["dst_pt2"])
                      - np.array(keypoints_dict["left_foot"]["dst_pt1"]))
            r_foot_v = (np.array(keypoints_dict["right_foot"]["dst_pt2"])
                      - np.array(keypoints_dict["right_foot"]["dst_pt1"]))

            if "left_foot" in links_dict:
                if ((l_foot_v[0] > 0 and links_dict["left_foot"]["direction"] == "l") or
                        (l_foot_v[0] < 0 and links_dict["left_foot"]["direction"] == "r")):
                    for part in ["left_thigh", "left_calf", "left_foot"]:
                        if part in links_dict:
                            links_dict[part] = self._flip_link(links_dict[part])

            if "right_foot" in links_dict:
                if ((r_foot_v[0] > 0 and links_dict["right_foot"]["direction"] == "l") or
                        (r_foot_v[0] < 0 and links_dict["right_foot"]["direction"] == "r")):
                    for part in ["right_thigh", "right_calf", "right_foot"]:
                        if part in links_dict:
                            links_dict[part] = self._flip_link(links_dict[part])

        return links_dict

    # ------------------------------------------------------------------
    # Keypoint parsing (mirrors parse_keypoints() in generate_silhouettes.py)
    # ------------------------------------------------------------------

    def _parse_keypoints(self, person_dict: dict):
        """
        Extract destination keypoints from a person dict (OpenPose 25-point format).

        Returns:
            (keypoints_dict, get_keypoint_fn) — the keypoints dict and the accessor
            function (needed by _get_head_direction in generate_silhouette).
        """
        pose_kp = person_dict.get("pose_keypoints_2d", [])
        keypoints = np.array(pose_kp).reshape((-1, 3)) if pose_kp else np.zeros((25, 3))
        num_kp = len(keypoints)

        left_hand_kp  = person_dict.get("hand_left_keypoints_2d", [])
        right_hand_kp = person_dict.get("hand_right_keypoints_2d", [])
        left_hand  = np.array(left_hand_kp).reshape((-1, 3))  if left_hand_kp  else np.zeros((0, 3))
        right_hand = np.array(right_hand_kp).reshape((-1, 3)) if right_hand_kp else np.zeros((0, 3))

        def get_keypoint(idx: int) -> np.ndarray:
            """Return (x, y, conf) for the given OpenPose 25-point index."""
            if idx < 0:
                return np.array([0.0, 0.0, 0.0])
            if idx < num_kp:
                return keypoints[idx]
            # mid_hip computed from hips if absent
            if idx == 8:
                l_hip = keypoints[12] if 12 < num_kp else np.zeros(3)
                r_hip = keypoints[9]  if  9 < num_kp else np.zeros(3)
                if l_hip[2] > 0 and r_hip[2] > 0:
                    return np.array([(l_hip[0]+r_hip[0])/2,
                                     (l_hip[1]+r_hip[1])/2, 1.0])
                return l_hip if l_hip[2] > 0 else r_hip
            fb = OPENPOSE_25_FALLBACK.get(idx, -1)
            if 0 <= fb < num_kp:
                return keypoints[fb]
            return np.zeros(3)

        keypoints_dict = {}
        head_middle = None

        for link_name, link_data in self.links_dict.items():
            if link_name == "neck":
                dst_pt1 = get_keypoint(link_data["pt1_idx"])[:2]
                if head_middle is None:
                    head_middle = self._get_head_middle_keypoint_mapped(get_keypoint)
                dst_pt2 = head_middle
            elif link_name == "head":
                if head_middle is None:
                    head_middle = self._get_head_middle_keypoint_mapped(get_keypoint)
                dst_pt1 = head_middle
                dst_pt2 = get_keypoint(link_data["pt2_idx"])[:2]  # nose
            elif link_name == "right_hand":
                dst_pt1 = right_hand[0][:2] if len(right_hand) > 0 else np.zeros(2)
                dst_pt2 = self._get_hand_middle_keypoint(right_hand)
            elif link_name == "left_hand":
                dst_pt1 = left_hand[0][:2] if len(left_hand) > 0 else np.zeros(2)
                dst_pt2 = self._get_hand_middle_keypoint(left_hand)
            else:
                dst_pt1 = get_keypoint(link_data["pt1_idx"])[:2]
                dst_pt2 = get_keypoint(link_data["pt2_idx"])[:2]

            keypoints_dict[link_name] = {
                "length":  np.linalg.norm(dst_pt2 - dst_pt1),
                "dst_pt1": dst_pt1,
                "dst_pt2": dst_pt2,
            }

        # Relative lengths normalised to torso
        if "torso" in keypoints_dict and keypoints_dict["torso"]["length"] > 0:
            torso_len = keypoints_dict["torso"]["length"]
            for kd in keypoints_dict.values():
                kd["relative_length"] = kd["length"] / torso_len

            # Adjust head length to match template proportions
            if ("head" in keypoints_dict and "head" in self.links_dict
                    and keypoints_dict["head"].get("relative_length", 0) > 0):
                head_len = keypoints_dict["head"]["length"]
                desired  = (self.links_dict["head"]["relative_length"]
                            / keypoints_dict["head"]["relative_length"]
                            * head_len)
                pt1 = keypoints_dict["head"]["dst_pt1"]
                pt2 = keypoints_dict["head"]["dst_pt2"]
                vec = pt2 - pt1
                norm = np.linalg.norm(vec)
                if norm > 0:
                    keypoints_dict["head"]["dst_pt2"] = pt1 + vec / norm * desired
                    keypoints_dict["head"]["length"] = desired
                    keypoints_dict["head"]["relative_length"] = desired / torso_len

        return keypoints_dict, get_keypoint

    # ------------------------------------------------------------------
    # Affine transformation
    # ------------------------------------------------------------------

    def _get_transformation_matrix(
        self,
        src_pt1: np.ndarray, src_pt2: np.ndarray,
        dst_pt1: np.ndarray, dst_pt2: np.ndarray,
        width_scale_coeff: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Compute affine 2×3 matrix to warp a body part template.
        Translate → rotate flat → scale → rotate to target → translate to dest.
        """
        src_vec = src_pt2 - src_pt1
        dst_vec = dst_pt2 - dst_pt1

        src_len = np.linalg.norm(src_vec)
        dst_len = np.linalg.norm(dst_vec)
        if src_len == 0 or dst_len == 0:
            return None

        src_angle = np.arctan2(src_vec[1], src_vec[0])
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

        t1 = np.array([[1, 0, -src_pt1[0]], [0, 1, -src_pt1[1]], [0, 0, 1]])
        r1 = np.array([[np.cos(-src_angle), -np.sin(-src_angle), 0],
                       [np.sin(-src_angle),  np.cos(-src_angle), 0],
                       [0, 0, 1]])
        s_l = dst_len / src_len
        s_w = s_l * width_scale_coeff
        s   = np.array([[s_l, 0, 0], [0, s_w, 0], [0, 0, 1]])
        r2  = np.array([[np.cos(dst_angle), -np.sin(dst_angle), 0],
                        [np.sin(dst_angle),  np.cos(dst_angle), 0],
                        [0, 0, 1]])
        t2  = np.array([[1, 0, dst_pt1[0]], [0, 1, dst_pt1[1]], [0, 0, 1]])

        return (t2 @ r2 @ s @ r1 @ t1)[:2, :]

    # ------------------------------------------------------------------
    # Main silhouette generation
    # ------------------------------------------------------------------

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
            person_dict:        OpenPose-format dict for one person.
            canvas_size:        (width, height) of the output canvas.
            parts_to_warp:      Body parts/groups to warp. None = all parts.
            width_scale_factor: Multiplier for limb thickness (default 0.85).

        Returns:
            PIL Image (RGBA) with warped body parts on a transparent background.
        """
        w, h = canvas_size

        # Determine active parts
        if parts_to_warp is None:
            active_parts = list(self.links_dict.keys())
        else:
            active_parts = expand_parts_list(parts_to_warp)

        # Z-order: neck → legs/feet → torso → arms → head → hands
        Z_ORDER = [
            "neck",
            "left_thigh", "right_thigh", "left_calf", "right_calf",
            "left_foot", "right_foot",
            "torso",
            "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
            "head",
            "left_hand", "right_hand",
        ]
        active_parts.sort(key=lambda p: Z_ORDER.index(p) if p in Z_ORDER else 999)

        # Parse keypoints
        keypoints_dict, get_keypoint = self._parse_keypoints(person_dict)

        # Work on a shallow copy so flips don't persist between calls
        links_dict = self.links_dict.copy()

        # Flip head/neck based on facing direction (mirrors generate_silhouettes.py)
        if "head" in links_dict:
            head_dir = self._get_head_direction(get_keypoint)
            if head_dir != links_dict["head"]["direction"]:
                links_dict["head"] = self._flip_link(links_dict["head"])
                if "neck" in links_dict:
                    links_dict["neck"] = self._flip_link(links_dict["neck"])

        # Flip arms / legs / torso based on pose geometry
        links_dict = self._adjust_link_direction(links_dict, keypoints_dict)

        # Transparent BGRA canvas
        canvas = np.zeros((h, w, 4), dtype=np.uint8)

        for link_name in active_parts:
            if link_name not in links_dict or link_name not in keypoints_dict:
                continue

            link_data = links_dict[link_name]
            kp_data   = keypoints_dict[link_name]

            if kp_data["length"] < 1:
                continue

            src_pt1 = link_data["pt1"]
            src_pt2 = link_data["pt2"]
            dst_pt1 = kp_data["dst_pt1"]
            dst_pt2 = kp_data["dst_pt2"]

            # Width scale coefficient
            width_scale_coeff = width_scale_factor
            if ("relative_length" in link_data and "relative_length" in kp_data
                    and kp_data["relative_length"] > 0):
                width_scale_coeff = (
                    link_data["relative_length"]
                    / kp_data["relative_length"]
                    * width_scale_factor
                )

            m = self._get_transformation_matrix(
                src_pt1, src_pt2, dst_pt1, dst_pt2, width_scale_coeff
            )
            if m is None:
                continue

            warped = cv2.warpAffine(link_data["image"], m, (w, h), flags=cv2.INTER_LINEAR)

            if warped.shape[2] == 4:
                link_alpha   = warped[:, :, 3:4] / 255.0
                link_bgr     = warped[:, :, :3]
                canvas_bgr   = canvas[:, :, :3]
                canvas_alpha = canvas[:, :, 3:4] / 255.0

                out_alpha      = link_alpha + canvas_alpha * (1 - link_alpha)
                out_alpha_safe = np.where(out_alpha == 0, 1, out_alpha)
                out_bgr        = (link_bgr * link_alpha
                                  + canvas_bgr * canvas_alpha * (1 - link_alpha)) / out_alpha_safe

                canvas[:, :, :3] = out_bgr.astype(np.uint8)
                canvas[:, :, 3]  = (out_alpha * 255).astype(np.uint8).squeeze()

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
