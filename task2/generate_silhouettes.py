import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_links_dir(links_dir):
    links_df = pd.read_csv(links_dir / "links.csv")
    links_dict = {}
    for _, row in links_df.iterrows():
        link_name = row["name"]
        link_data = {}
        link_data["pt1_idx"] = row["pt1_idx"]
        link_data["pt2_idx"] = row["pt2_idx"]
        link_data["pt1"] = np.array([row["pt1_x"], row["pt1_y"]])
        link_data["pt2"] = np.array([row["pt2_x"], row["pt2_y"]])
        link_img_path = links_dir / f"{link_name}.png"
        link_data["image"] = cv2.imread(str(link_img_path), cv2.IMREAD_UNCHANGED)
        link_data["length"] = np.linalg.norm(link_data["pt2"] - link_data["pt1"])
        link_data["direction"] = row["direction"]  # l or r
        links_dict[link_name] = link_data

    torso_length = links_dict["torso"]["length"]

    for link_name, link_data in links_dict.items():
        links_dict[link_name]["relative_length"] = link_data["length"] / torso_length

    return links_dict


def get_hand_middle_keypoint(hand_keypoints):
    idxes = [1, 5, 9, 13, 17]
    visible_keypoints = [hand_keypoints[i] for i in idxes if hand_keypoints[i][2] != 0]
    visible_keypoints = np.array(visible_keypoints)
    hand_middle_keypoint = np.mean(visible_keypoints[:, :2], axis=0)
    return hand_middle_keypoint


def get_head_middle_keypoint(keypoints):
    r_ear_keypoint = keypoints[17]
    l_ear_keypoint = keypoints[18]

    if r_ear_keypoint[2] != 0 and l_ear_keypoint[2] != 0:
        head_middle_keypoint = (r_ear_keypoint[:2] + l_ear_keypoint[:2]) / 2
    elif r_ear_keypoint[2] != 0:
        head_middle_keypoint = r_ear_keypoint[:2]
    else:
        head_middle_keypoint = l_ear_keypoint[:2]

    return head_middle_keypoint


def parse_keypoints(keypoints, left_hand_keypoints, right_hand_keypoints, links_dict):
    keypoints_dict = {}
    head_middle_keypoint = None
    for link_name, link_data in links_dict.items():
        # Calculate the head middle keypoint based on the visible ear keypoints
        if link_name == "neck":
            dst_pt1 = keypoints[link_data["pt1_idx"]][:2]
            if head_middle_keypoint is None:
                head_middle_keypoint = get_head_middle_keypoint(keypoints)
            dst_pt2 = head_middle_keypoint
        elif link_name == "head":
            if head_middle_keypoint is None:
                head_middle_keypoint = get_head_middle_keypoint(keypoints)
            dst_pt1 = head_middle_keypoint
            dst_pt2 = keypoints[link_data["pt2_idx"]][:2]
        elif link_name == "right_hand":
            dst_pt1 = right_hand_keypoints[0][:2]
            dst_pt2 = get_hand_middle_keypoint(right_hand_keypoints)
        elif link_name == "left_hand":
            dst_pt1 = left_hand_keypoints[0][:2]
            dst_pt2 = get_hand_middle_keypoint(left_hand_keypoints)
        else:
            dst_pt1 = keypoints[link_data["pt1_idx"]][:2]
            dst_pt2 = keypoints[link_data["pt2_idx"]][:2]

        length = np.linalg.norm(dst_pt2 - dst_pt1)
        keypoints_dict[link_name] = {}
        keypoints_dict[link_name]["length"] = length
        keypoints_dict[link_name]["dst_pt1"] = dst_pt1
        keypoints_dict[link_name]["dst_pt2"] = dst_pt2

    torso_length = keypoints_dict["torso"]["length"]

    for link_name, keypoint_link_data in keypoints_dict.items():
        keypoint_link_data["relative_length"] = (
            keypoint_link_data["length"] / torso_length
        )

    # Adjust the head dst point so that the relative lengths match
    head_length = keypoints_dict["head"]["length"]
    desired_head_length = (
        links_dict["head"]["relative_length"]
        / keypoints_dict["head"]["relative_length"]
        * head_length
    )
    dst_pt1 = keypoints_dict["head"]["dst_pt1"]
    dst_pt2 = keypoints_dict["head"]["dst_pt2"]
    head_vec = dst_pt2 - dst_pt1
    head_vec_normalized = head_vec / np.linalg.norm(head_vec)
    dst_pt2 = dst_pt1 + head_vec_normalized * desired_head_length
    keypoints_dict["head"]["length"] = desired_head_length
    keypoints_dict["head"]["dst_pt2"] = dst_pt2
    keypoints_dict["head"]["relative_length"] = (
        keypoints_dict["head"]["length"] / torso_length
    )

    return keypoints_dict


def get_head_direction(keypoints):
    r_ear_keypoint = keypoints[17]
    l_ear_keypoint = keypoints[18]
    nose_keypoint = keypoints[0][:2]
    # Calculate how far the nose is on the line connecting the ears
    # Assume persons are not facing in the direction opposite to the camera
    # If the nose is closer than middle to the right ear, we say the head is facing left, otherwise right
    if r_ear_keypoint[2] != 0 and l_ear_keypoint[2] != 0:
        ear_vec = l_ear_keypoint[:2] - r_ear_keypoint[:2]
        nose_vec = nose_keypoint - r_ear_keypoint[:2]
        proj_length = np.dot(nose_vec, ear_vec) / np.dot(ear_vec, ear_vec)
        if proj_length < 0.5:
            return "l"
        else:
            return "r"
    elif r_ear_keypoint[2] != 0:
        return "r"
    else:
        return "l"


def flip_link(link_data):
    flipped_link = link_data.copy()
    flipped_image = cv2.flip(link_data["image"], 1)
    flipped_link["image"] = flipped_image
    flipped_link["pt1"] = np.array(
        [link_data["image"].shape[1] - link_data["pt1"][0], link_data["pt1"][1]]
    )
    flipped_link["pt2"] = np.array(
        [link_data["image"].shape[1] - link_data["pt2"][0], link_data["pt2"][1]]
    )
    return flipped_link


def adjust_link_direction(links_dict, keypoints_dict):
    # Process arms
    # Assume image y as x and image x as y for angle calculation
    right_upper_arm_vec = np.array(
        keypoints_dict["right_upper_arm"]["dst_pt1"]
    ) - np.array(keypoints_dict["right_upper_arm"]["dst_pt2"])
    right_forearm_vec = np.array(keypoints_dict["right_forearm"]["dst_pt2"]) - np.array(
        keypoints_dict["right_forearm"]["dst_pt1"]
    )
    left_upper_arm_vec = np.array(
        keypoints_dict["left_upper_arm"]["dst_pt1"]
    ) - np.array(keypoints_dict["left_upper_arm"]["dst_pt2"])
    left_forearm_vec = np.array(keypoints_dict["left_forearm"]["dst_pt2"]) - np.array(
        keypoints_dict["left_forearm"]["dst_pt1"]
    )

    right_upper_arm_angle = np.degrees(
        (np.arctan2(right_upper_arm_vec[0], right_upper_arm_vec[1]))
    )
    right_forearm_angle = np.degrees(
        (np.arctan2(right_forearm_vec[0], right_forearm_vec[1]))
    )
    left_upper_arm_angle = np.degrees(
        (np.arctan2(left_upper_arm_vec[0], left_upper_arm_vec[1]))
    )
    left_forearm_angle = np.degrees(
        (np.arctan2(left_forearm_vec[0], left_forearm_vec[1]))
    )

    right_elbow_angle = (right_upper_arm_angle - right_forearm_angle) % 360
    left_elbow_angle = (left_upper_arm_angle - left_forearm_angle) % 360

    if (
        right_elbow_angle > 180 and links_dict["right_forearm"]["direction"] == "r"
    ) or (right_elbow_angle < 180 and links_dict["right_forearm"]["direction"] == "l"):
        links_dict["right_upper_arm"] = flip_link(links_dict["right_upper_arm"])
        links_dict["right_forearm"] = flip_link(links_dict["right_forearm"])
        links_dict["right_hand"] = flip_link(links_dict["right_hand"])

    if (left_elbow_angle > 180 and links_dict["left_forearm"]["direction"] == "r") or (
        left_elbow_angle < 180 and links_dict["left_forearm"]["direction"] == "l"
    ):
        links_dict["left_upper_arm"] = flip_link(links_dict["left_upper_arm"])
        links_dict["left_forearm"] = flip_link(links_dict["left_forearm"])
        links_dict["left_hand"] = flip_link(links_dict["left_hand"])

    # Process torso
    if (
        (right_elbow_angle > 180 and left_elbow_angle > 180)
        and links_dict["torso"]["direction"] == "r"
        or (right_elbow_angle < 180 and left_elbow_angle < 180)
        and links_dict["torso"]["direction"] == "l"
    ):
        links_dict["torso"] = flip_link(links_dict["torso"])

    # Process legs
    left_foot_vec = np.array(keypoints_dict["left_foot"]["dst_pt2"]) - np.array(
        keypoints_dict["left_foot"]["dst_pt1"]
    )
    right_foot_vec = np.array(keypoints_dict["right_foot"]["dst_pt2"]) - np.array(
        keypoints_dict["right_foot"]["dst_pt1"]
    )

    if (left_foot_vec[0] > 0 and links_dict["left_foot"]["direction"] == "l") or (
        left_foot_vec[0] < 0 and links_dict["left_foot"]["direction"] == "r"
    ):
        links_dict["left_thigh"] = flip_link(links_dict["left_thigh"])
        links_dict["left_calf"] = flip_link(links_dict["left_calf"])
        links_dict["left_foot"] = flip_link(links_dict["left_foot"])

    if (right_foot_vec[0] > 0 and links_dict["right_foot"]["direction"] == "l") or (
        right_foot_vec[0] < 0 and links_dict["right_foot"]["direction"] == "r"
    ):
        links_dict["right_thigh"] = flip_link(links_dict["right_thigh"])
        links_dict["right_calf"] = flip_link(links_dict["right_calf"])
        links_dict["right_foot"] = flip_link(links_dict["right_foot"])

    return links_dict


def get_link_transformation_matrix(
    src_pt1, src_pt2, dst_pt1, dst_pt2, width_scale_coeff=1.0
):
    src_vec = src_pt2 - src_pt1
    dst_vec = dst_pt2 - dst_pt1

    src_len = np.linalg.norm(src_vec)
    dst_len = np.linalg.norm(dst_vec)

    src_angle = np.arctan2(src_vec[1], src_vec[0])
    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])

    # 1. Move src_pt1 to (0,0)
    t1 = np.array([[1, 0, -src_pt1[0]], [0, 1, -src_pt1[1]], [0, 0, 1]])

    # 2. Rotate the source link so it is flat on the x axis (0 degrees)
    r1 = np.array(
        [
            [np.cos(-src_angle), -np.sin(-src_angle), 0],
            [np.sin(-src_angle), np.cos(-src_angle), 0],
            [0, 0, 1],
        ]
    )

    # 3. Apply scaling
    # Length scale s_l goes on x because we rotated the link to the x axis
    # Width scale s_w goes on y because y is now the thickness
    s_l = dst_len / src_len
    s_w = s_l * width_scale_coeff

    s = np.array([[s_l, 0, 0], [0, s_w, 0], [0, 0, 1]])

    # 4. Rotate to the target angle on the background
    r2 = np.array(
        [
            [np.cos(dst_angle), -np.sin(dst_angle), 0],
            [np.sin(dst_angle), np.cos(dst_angle), 0],
            [0, 0, 1],
        ]
    )

    # 5. Move to dst_pt1
    t2 = np.array([[1, 0, dst_pt1[0]], [0, 1, dst_pt1[1]], [0, 0, 1]])

    # Full matrix: t2 * r2 * s * r1 * t1
    m = t2 @ r2 @ s @ r1 @ t1

    # Return 2x3
    return m[:2, :]


def generate_silhouette(
    persons_dict, links_dict_src, background_img, width_scale_factor=0.85
):
    links_dict = links_dict_src.copy()

    keypoints = persons_dict["pose_keypoints_2d"]
    keypoints = np.array(keypoints).reshape((-1, 3))

    left_hand_keypoints = persons_dict["hand_left_keypoints_2d"]
    left_hand_keypoints = np.array(left_hand_keypoints).reshape((-1, 3))

    right_hand_keypoints = persons_dict["hand_right_keypoints_2d"]
    right_hand_keypoints = np.array(right_hand_keypoints).reshape((-1, 3))

    keypoints_dict = parse_keypoints(
        keypoints, left_hand_keypoints, right_hand_keypoints, links_dict
    )

    # Adjust link direction if the link and keypoint directions do not match
    head_direction = get_head_direction(keypoints)
    if head_direction != links_dict["head"]["direction"]:
        links_dict["head"] = flip_link(links_dict["head"])
        links_dict["neck"] = flip_link(links_dict["neck"])
    links_dict = adjust_link_direction(links_dict, keypoints_dict)

    h, w = background_img.shape[:2]

    canvas = background_img.copy()

    for link_name, link_data in links_dict.items():
        src_pt1 = link_data["pt1"]
        src_pt2 = link_data["pt2"]
        link_img = link_data["image"]

        dst_pt1 = keypoints_dict[link_name]["dst_pt1"]
        dst_pt2 = keypoints_dict[link_name]["dst_pt2"]

        width_scale_coeff = (
            link_data["relative_length"]
            / keypoints_dict[link_name]["relative_length"]
            * width_scale_factor
        )

        m = get_link_transformation_matrix(
            src_pt1, src_pt2, dst_pt1, dst_pt2, width_scale_coeff
        )

        warped_link = cv2.warpAffine(
            link_img,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
        )
        link_bgr = warped_link[:, :, :3]
        link_alpha = warped_link[:, :, 3:4] / 255.0

        canvas = (1 - link_alpha) * canvas + link_alpha * link_bgr

    return canvas


def main():
    openpose_dir = Path("data/openpose")
    keypoints_path = openpose_dir / "keypoints.json"

    with open(keypoints_path, "r") as f:
        keypoints_json = json.load(f)

    links_dir = Path("data/Prince_Achmed/links")
    links_dict = parse_links_dir(links_dir)

    background_img_path = openpose_dir / "background.png"
    background_img = cv2.imread(str(background_img_path))

    output_dir = Path("task2_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "silhouette_output.png"

    width_scale_factor = 0.85

    for persons_dict in keypoints_json["people"]:
        person_id = persons_dict["person_id"]
        print(f"Processing person with person_id {person_id}")
        background_img = generate_silhouette(
            persons_dict, links_dict, background_img, width_scale_factor
        )

    cv2.imwrite(str(output_path), background_img)


if __name__ == "__main__":
    main()
