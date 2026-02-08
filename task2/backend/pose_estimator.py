"""
MediaPipe Pose Estimator Module

Uses MediaPipe Pose Landmarker (Tasks API) for extracting keypoints 
and converts to OpenPose 25-point format.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import logging
import os
import urllib.request

logger = logging.getLogger("PoseEstimator")

# MediaPipe Pose 33 landmarks to OpenPose 25 keypoint mapping
# MediaPipe landmarks:
# 0: nose, 1: left_eye_inner, 2: left_eye, 3: left_eye_outer
# 4: right_eye_inner, 5: right_eye, 6: right_eye_outer
# 7: left_ear, 8: right_ear, 9: mouth_left, 10: mouth_right
# 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow
# 15: left_wrist, 16: right_wrist, 17-22: hands (pinky, index, thumb)
# 23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee
# 27: left_ankle, 28: right_ankle, 29: left_heel, 30: right_heel
# 31: left_foot_index, 32: right_foot_index

# OpenPose 25 keypoint indices:
# 0: nose, 1: neck (avg of shoulders), 2: right_shoulder, 3: right_elbow, 4: right_wrist
# 5: left_shoulder, 6: left_elbow, 7: left_wrist, 8: mid_hip (avg)
# 9: right_hip, 10: right_knee, 11: right_ankle
# 12: left_hip, 13: left_knee, 14: left_ankle
# 15: right_eye, 16: left_eye, 17: right_ear, 18: left_ear
# 19: left_big_toe, 20: left_small_toe, 21: left_heel
# 22: right_big_toe, 23: right_small_toe, 24: right_heel

MEDIAPIPE_TO_OPENPOSE = {
    0: 0,    # nose -> nose
    # 1 is neck - computed from shoulders
    2: 12,   # right_shoulder <- MP right_shoulder
    3: 14,   # right_elbow <- MP right_elbow
    4: 16,   # right_wrist <- MP right_wrist
    5: 11,   # left_shoulder <- MP left_shoulder
    6: 13,   # left_elbow <- MP left_elbow
    7: 15,   # left_wrist <- MP left_wrist
    # 8 is mid_hip - computed from hips
    9: 24,   # right_hip <- MP right_hip
    10: 26,  # right_knee <- MP right_knee
    11: 28,  # right_ankle <- MP right_ankle
    12: 23,  # left_hip <- MP left_hip
    13: 25,  # left_knee <- MP left_knee
    14: 27,  # left_ankle <- MP left_ankle
    15: 5,   # right_eye <- MP right_eye
    16: 2,   # left_eye <- MP left_eye
    17: 8,   # right_ear <- MP right_ear
    18: 7,   # left_ear <- MP left_ear
    19: 31,  # left_big_toe <- MP left_foot_index
    20: 31,  # left_small_toe <- MP left_foot_index
    21: 29,  # left_heel <- MP left_heel
    22: 32,  # right_big_toe <- MP right_foot_index
    23: 32,  # right_small_toe <- MP right_foot_index
    24: 30,  # right_heel <- MP right_heel
}

# Model URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_heavy.task"

# Hand Model
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


class PoseEstimator:
    """
    Pose estimator using MediaPipe Pose Landmarker for detecting body keypoints.
    Outputs keypoints in OpenPose 25-point JSON format.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the pose estimator."""
        self.device = device
        self._landmarker = None
        self._hand_landmarker = None
        self._initialized = False

    def _download_models(self):
        """Download pose and hand landmarker models if not present."""
        # Pose Model
        if not MODEL_PATH.exists():
            logger.info(f"Downloading MediaPipe Pose model to {MODEL_PATH}...")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        # Hand Model
        if not HAND_MODEL_PATH.exists():
            logger.info(f"Downloading MediaPipe Hand model to {HAND_MODEL_PATH}...")
            HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        
        return str(MODEL_PATH), str(HAND_MODEL_PATH)

    def _lazy_init(self):
        """Lazily initialize MediaPipe Pose Landmarker on first use."""
        if self._initialized:
            return

        logger.info("Initializing MediaPipe Pose Landmarker...")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download models if needed
            pose_model_path, hand_model_path = self._download_models()
            
            # Create pose landmarker options
            base_options = python.BaseOptions(model_asset_path=pose_model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                num_poses=5  # Support up to 5 people
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

            # Create hand landmarker options
            hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
            hand_options = vision.HandLandmarkerOptions(
                base_options=hand_base_options,
                num_hands=10,  # Support up to 10 hands (5 people)
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

            self._mp = mp  # Save reference
            self._initialized = True
            logger.info("MediaPipe Pose and Hand Landmarkers initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose Landmarker: {e}")
            import traceback
            traceback.print_exc()
            self._landmarker = None
            self._initialized = True

    def detect_poses(
        self,
        image: Union[np.ndarray, str, Image.Image],
        include_hands: bool = True,
        include_face: bool = False,
    ) -> dict:
        """
        Detect poses in an image.
        
        Returns:
            Dict in OpenPose JSON format with 25 body keypoints
        """
        self._lazy_init()

        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            import cv2
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        width, height = image.size
        image_rgb = np.array(image)

        if self._landmarker is None:
            logger.warning("MediaPipe Pose not available, returning empty keypoints")
            return self._create_empty_result(height, width)

        try:
            # Create MediaPipe Image
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
            
            # Run pose detection
            pose_result = self._landmarker.detect(mp_image)
            
            # Run hand detection
            if include_hands and self._hand_landmarker:
                hand_result = self._hand_landmarker.detect(mp_image)
            else:
                hand_result = None
            
            if not pose_result.pose_landmarks:
                logger.warning("No pose detected in image")
                return self._create_empty_result(height, width)
            
            people_list = []
            
            # Helper to find which person a hand belongs to
            # Associating hand with person based on wrist distance
            def find_closest_person_idx(hand_kps, people_wrists):
                hand_center = np.mean([kp[:2] for kp in hand_kps], axis=0)
                min_dist = float('inf')
                closest_idx = -1
                
                for idx, wrists in enumerate(people_wrists):
                    # Check distance to left wrist (idx 4 / MP 15) and right wrist (idx 7 / MP 16)
                    # Wrist indices in OpenPose are 4 (right) and 7 (left)
                    # But here we are matching detected hand against the wrist position from the body pose
                    
                    # Left wrist check
                    if wrists['left']:
                        dist = np.linalg.norm(hand_center - wrists['left'][:2])
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                            
                    # Right wrist check
                    if wrists['right']:
                        dist = np.linalg.norm(hand_center - wrists['right'][:2])
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                            
                # Reject if too far (e.g. > 20% of image diag)
                if min_dist > (width**2 + height**2)**0.5 * 0.2:
                    return -1
                return closest_idx

            # 1. Process all bodies first
            for i, mp_landmarks in enumerate(pose_result.pose_landmarks):
                openpose_keypoints = self._convert_to_openpose25(mp_landmarks, width, height)
                
                people_list.append({
                    "pose_keypoints_2d": openpose_keypoints,
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "face_keypoints_2d": [],
                    # Internal helper data for matching
                    "_wrists": {
                        "left": self._get_kp_from_list(openpose_keypoints, 7),  # Left wrist
                        "right": self._get_kp_from_list(openpose_keypoints, 4)  # Right wrist
                    }
                })

            # 2. Process hands and assign to people
            if hand_result and hand_result.hand_landmarks:
                # Pre-calculate wrist positions for all people
                people_wrists = [p["_wrists"] for p in people_list]
                
                for col_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    handedness = hand_result.handedness[col_idx][0]
                    # MediaPipe says "Left" for left hand, "Right" for right hand
                    is_left = (handedness.category_name == "Left")
                    
                    kps = self._convert_hand_to_openpose21(hand_landmarks, width, height)
                    
                    # Find which person this hand belongs to
                    # We check distance to the relevant wrist
                    closest_person_idx = -1
                    min_dist = float('inf')
                    
                    hand_wrist_pt = np.array(kps[0:2]) # First point is wrist
                    
                    for p_idx, p in enumerate(people_list):
                        # Get the corresponding wrist from the body pose
                        # If hand is "Left", match with Left Wrist (keypoint 7)
                        # If hand is "Right", match with Right Wrist (keypoint 4)
                        body_wrist = p["_wrists"]["left"] if is_left else p["_wrists"]["right"]
                        
                        if body_wrist is not None and body_wrist[2] > 0.1: # Visible
                            dist = np.linalg.norm(hand_wrist_pt - body_wrist[:2])
                            if dist < min_dist:
                                min_dist = dist
                                closest_person_idx = p_idx
                    
                    # Accept match if within reasonable distance (e.g. < 10% of image size)
                    limit = max(width, height) * 0.15
                    if closest_person_idx != -1 and min_dist < limit:
                        target_person = people_list[closest_person_idx]
                        if is_left:
                            target_person["hand_left_keypoints_2d"] = kps
                        else:
                            target_person["hand_right_keypoints_2d"] = kps

            # Clean up internal data
            for p in people_list:
                del p["_wrists"]

            return {
                "people": people_list,
                "canvas_height": height,
                "canvas_width": width,
            }

        except Exception as e:
            logger.error(f"Pose detection failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_result(height, width)
            
    def draw_poses(self, image: Image.Image, pose_result: dict) -> Image.Image:
        """Visualize detected keypoints on the image."""
        import cv2
        vis_img = np.array(image.convert("RGB"))
        
        # OpenPose 25 Skeleton connections
        # (p1, p2) indices
        skeleton = [
            (0, 1), (0, 15), (0, 16), (15, 17), (16, 18), # Head
            (1, 2), (1, 5), (1, 8), # Torso/Shoulders/Neck
            (2, 3), (3, 4), # Right Arm
            (5, 6), (6, 7), # Left Arm
            (8, 9), (8, 12), # Hips
            (9, 10), (10, 11), # Right Leg
            (12, 13), (13, 14), # Left Leg
            (11, 24), (11, 22), (22, 23), # Right Foot
            (14, 21), (14, 19), (19, 20)  # Left Foot
        ]
        
        people = pose_result.get("people", [])
        for person in people:
            kps = np.array(person.get("pose_keypoints_2d", [])).reshape(-1, 3)
            # Draw Skeleton
            for p1, p2 in skeleton:
                if p1 < len(kps) and p2 < len(kps):
                    kp1 = kps[p1]
                    kp2 = kps[p2]
                    if kp1[2] > 0.1 and kp2[2] > 0.1:
                        pt1 = (int(kp1[0]), int(kp1[1]))
                        pt2 = (int(kp2[0]), int(kp2[1]))
                        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
            
            # Draw Points
            for i, kp in enumerate(kps):
                if kp[2] > 0.1:
                    pt = (int(kp[0]), int(kp[1]))
                    cv2.circle(vis_img, pt, 3, (0, 0, 255), -1)

            # Draw Hands (Wrists connected to hand points if available)
            # Simple visualization for hands just to show they are detected
            for hand_key in ["hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                hand_kps = person.get(hand_key, [])
                if hand_kps:
                    h_kps = np.array(hand_kps).reshape(-1, 3)
                    for kp in h_kps:
                         if kp[2] > 0.1:
                            cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)

        return Image.fromarray(vis_img)
            
    def _get_kp_from_list(self, keypoints_list, idx):
        """Helper to get (x,y,conf) from flat flattened list [x1,y1,c1, x2,y2,c2...]"""
        if idx * 3 + 2 < len(keypoints_list):
            return np.array(keypoints_list[idx*3 : idx*3+3])
        return None

    def _convert_hand_to_openpose21(self, landmarks, width: int, height: int) -> list:
        """Convert MediaPipe hand landmarks to OpenPose 21 keypoint format."""
        # MediaPipe Hand landmarks (0-20) map 1:1 to OpenPose Hand keypoints (0-20)
        # 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
        keypoints = []
        for lm in landmarks:
            keypoints.extend([lm.x * width, lm.y * height, 1.0])  # Visibility 1.0
        return keypoints

    def _convert_to_openpose25(self, mp_landmarks, width: int, height: int) -> list:
        """Convert MediaPipe 33 landmarks to OpenPose 25 keypoint format."""
        keypoints_25 = []
        
        def get_mp_kp(mp_idx):
            lm = mp_landmarks[mp_idx]
            return [lm.x * width, lm.y * height, lm.visibility if hasattr(lm, 'visibility') else 1.0]
        
        for op_idx in range(25):
            if op_idx == 0:  # nose
                kp = get_mp_kp(0)
            elif op_idx == 1:  # neck (average of shoulders)
                l_sh = get_mp_kp(11)
                r_sh = get_mp_kp(12)
                kp = [(l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2, 
                      min(l_sh[2], r_sh[2])]
            elif op_idx == 8:  # mid_hip (average of hips)
                l_hip = get_mp_kp(23)
                r_hip = get_mp_kp(24)
                kp = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2,
                      min(l_hip[2], r_hip[2])]
            elif op_idx in MEDIAPIPE_TO_OPENPOSE:
                mp_idx = MEDIAPIPE_TO_OPENPOSE[op_idx]
                kp = get_mp_kp(mp_idx)
            else:
                kp = [0, 0, 0]
            
            keypoints_25.extend(kp)
        
        return keypoints_25

    def _create_empty_result(self, height: int, width: int) -> dict:
        """Create empty result when detection fails."""
        return {
            "people": [],
            "canvas_height": height,
            "canvas_width": width,
        }


# Module-level singleton
_pose_estimator: Optional[PoseEstimator] = None


def get_pose_estimator() -> PoseEstimator:
    """Get or create the global pose estimator instance."""
    global _pose_estimator
    if _pose_estimator is None:
        _pose_estimator = PoseEstimator()
    return _pose_estimator
