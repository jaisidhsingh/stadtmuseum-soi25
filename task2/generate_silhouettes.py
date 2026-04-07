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
    hand_middle_keypoint = np.mean(visible_keypoints, axis=0)
    return hand_middle_keypoint


def get_head_middle_keypoint(keypoints):
    r_ear_keypoint = keypoints[17]
    l_ear_keypoint = keypoints[18]

    if r_ear_keypoint[2] != 0 and l_ear_keypoint[2] != 0:
        head_middle_keypoint = (r_ear_keypoint + l_ear_keypoint) / 2
    elif r_ear_keypoint[2] != 0:
        head_middle_keypoint = r_ear_keypoint
    else:
        head_middle_keypoint = l_ear_keypoint

    return head_middle_keypoint


def get_hat_keypoints(head_dict, offset_const=0.8):
    dst_pt1 = head_dict["dst_pt1"]
    dst_pt2 = head_dict["dst_pt2"]

    dx = dst_pt2[0] - dst_pt1[0]
    dy = dst_pt2[1] - dst_pt1[1]

    perp_dx = dy
    perp_dy = -dx

    length = np.sqrt(perp_dx**2 + perp_dy**2)
    perp_dx /= length
    perp_dy /= length

    hat_offset_dist = length * offset_const

    hat_pt1 = np.array(
        [
            dst_pt1[0] + perp_dx * hat_offset_dist,
            dst_pt1[1] + perp_dy * hat_offset_dist,
        ]
    )

    hat_pt2 = np.array(
        [
            dst_pt2[0] + perp_dx * hat_offset_dist,
            dst_pt2[1] + perp_dy * hat_offset_dist,
        ]
    )

    hat_dict = {
        "dst_pt1": hat_pt1,
        "dst_pt2": hat_pt2,
        "length": np.linalg.norm(hat_pt2 - hat_pt1),
    }
    return hat_dict


def parse_keypoints(
    keypoints, left_hand_keypoints, right_hand_keypoints, links_dict, add_hat=False
):
    keypoints_dict = {}
    head_middle_keypoint = None
    for link_name, link_data in links_dict.items():
        # Calculate the head middle keypoint based on the visible ear keypoints
        if link_name == "neck":
            dst_pt1 = keypoints[link_data["pt1_idx"]]
            if head_middle_keypoint is None:
                head_middle_keypoint = get_head_middle_keypoint(keypoints)
            dst_pt2 = head_middle_keypoint
        elif link_name == "head":
            if head_middle_keypoint is None:
                head_middle_keypoint = get_head_middle_keypoint(keypoints)
            dst_pt1 = head_middle_keypoint

            r_ear_keypoint = keypoints[17]
            l_ear_keypoint = keypoints[18]
            nose_keypoint = keypoints[0]

            # If nose visible, person not facing away from the camera
            if nose_keypoint[2] != 0:
                # If only one ear is visible, use nose keypoint as dst_pt2
                if r_ear_keypoint[2] == 0 or l_ear_keypoint[2] == 0:
                    dst_pt2 = keypoints[link_data["pt2_idx"]]
                # Otherwise, use the ear keypoint in the direction the head is facing
                else:
                    head_direction = get_head_direction(keypoints)
                    if head_direction == "r":
                        dst_pt2 = l_ear_keypoint
                    else:
                        dst_pt2 = r_ear_keypoint
            # TODO Implement logic for when the person is facing away from the camera
            # Otherwise, person is facing away from the camera, defaulting to right ear keypoint
            else:
                dst_pt2 = r_ear_keypoint

        elif link_name == "right_hand":
            dst_pt1 = right_hand_keypoints[0]
            dst_pt2 = get_hand_middle_keypoint(right_hand_keypoints)
        elif link_name == "left_hand":
            dst_pt1 = left_hand_keypoints[0]
            dst_pt2 = get_hand_middle_keypoint(left_hand_keypoints)
        else:
            dst_pt1 = keypoints[link_data["pt1_idx"]]
            dst_pt2 = keypoints[link_data["pt2_idx"]]

        # Skip links with keypoint visibility 0
        if dst_pt1[2] == 0 or dst_pt2[2] == 0:
            continue

        dst_pt1 = dst_pt1[:2]
        dst_pt2 = dst_pt2[:2]

        length = np.linalg.norm(dst_pt2 - dst_pt1)
        keypoints_dict[link_name] = {}
        keypoints_dict[link_name]["length"] = length
        keypoints_dict[link_name]["dst_pt1"] = dst_pt1
        keypoints_dict[link_name]["dst_pt2"] = dst_pt2

    if add_hat and "head" in keypoints_dict:
        keypoints_dict["hat"] = get_hat_keypoints(keypoints_dict["head"])

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
    # TODO Implement logic for when the person is facing away from the camera
    if keypoints[0][2] == 0:
        return "r"

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


def get_angle_between_links(link1_vec, link2_vec):
    # Assume image y as x and image x as y for angle calculation
    angle1 = np.degrees(np.arctan2(link1_vec[0], link1_vec[1]))
    angle2 = np.degrees(np.arctan2(link2_vec[0], link2_vec[1]))
    angle = (angle1 - angle2) % 360
    return angle


def adjust_arm(links_dict, keypoints_dict, arm="right"):
    elbow_angle = None
    upper_arm_key = f"{arm}_upper_arm"
    forearm_key = f"{arm}_forearm"

    if upper_arm_key in keypoints_dict and forearm_key in keypoints_dict:
        upper_arm_vec = (
            keypoints_dict[upper_arm_key]["dst_pt1"]
            - keypoints_dict[upper_arm_key]["dst_pt2"]
        )
        forearm_vec = (
            keypoints_dict[forearm_key]["dst_pt2"]
            - keypoints_dict[forearm_key]["dst_pt1"]
        )

        elbow_angle = get_angle_between_links(upper_arm_vec, forearm_vec)

        if (elbow_angle > 180 and links_dict[forearm_key]["direction"] == "r") or (
            elbow_angle < 180 and links_dict[forearm_key]["direction"] == "l"
        ):
            links_dict[upper_arm_key] = flip_link(links_dict[upper_arm_key])
            links_dict[forearm_key] = flip_link(links_dict[forearm_key])
            if f"{arm}_hand" in links_dict:
                links_dict[f"{arm}_hand"] = flip_link(links_dict[f"{arm}_hand"])

    return elbow_angle


def adjust_leg(links_dict, keypoints_dict, leg="right"):
    flip = False

    if f"{leg}_foot" in keypoints_dict:
        foot_vec = np.array(keypoints_dict[f"{leg}_foot"]["dst_pt2"]) - np.array(
            keypoints_dict[f"{leg}_foot"]["dst_pt1"]
        )
        if f"{leg}_calf" in keypoints_dict:
            calf_vec = np.array(keypoints_dict[f"{leg}_calf"]["dst_pt1"]) - np.array(
                keypoints_dict[f"{leg}_calf"]["dst_pt2"]
            )
            heel_angle = get_angle_between_links(calf_vec, foot_vec)
            if (heel_angle > 180 and links_dict[f"{leg}_foot"]["direction"] == "r") or (
                heel_angle < 180 and links_dict[f"{leg}_foot"]["direction"] == "l"
            ):
                flip = True
        elif (foot_vec[0] > 0 and links_dict[f"{leg}_foot"]["direction"] == "l") or (
            foot_vec[0] < 0 and links_dict[f"{leg}_foot"]["direction"] == "r"
        ):
            flip = True

    if flip:
        links_dict[f"{leg}_foot"] = flip_link(links_dict[f"{leg}_foot"])
        if f"{leg}_thigh" in links_dict:
            links_dict[f"{leg}_thigh"] = flip_link(links_dict[f"{leg}_thigh"])
        if f"{leg}_calf" in links_dict:
            links_dict[f"{leg}_calf"] = flip_link(links_dict[f"{leg}_calf"])


def adjust_link_direction(links_dict, keypoints_dict):
    # Process arms
    right_elbow_angle = adjust_arm(links_dict, keypoints_dict, "right")
    left_elbow_angle = adjust_arm(links_dict, keypoints_dict, "left")

    # Process torso
    if right_elbow_angle is not None and left_elbow_angle is not None:
        if (
            (right_elbow_angle > 180 and left_elbow_angle > 180)
            and links_dict["torso"]["direction"] == "r"
            or (right_elbow_angle < 180 and left_elbow_angle < 180)
            and links_dict["torso"]["direction"] == "l"
        ):
            links_dict["torso"] = flip_link(links_dict["torso"])

    # Process legs
    adjust_leg(links_dict, keypoints_dict, "right")
    adjust_leg(links_dict, keypoints_dict, "left")

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
    persons_dict, links_dict_src, background_img, width_scale_factor=0.85, add_hat=False
):
    links_dict = links_dict_src.copy()

    keypoints = persons_dict["pose_keypoints_2d"]
    keypoints = np.array(keypoints).reshape((-1, 3))

    left_hand_keypoints = persons_dict["hand_left_keypoints_2d"]
    left_hand_keypoints = np.array(left_hand_keypoints).reshape((-1, 3))

    right_hand_keypoints = persons_dict["hand_right_keypoints_2d"]
    right_hand_keypoints = np.array(right_hand_keypoints).reshape((-1, 3))

    keypoints_dict = parse_keypoints(
        keypoints, left_hand_keypoints, right_hand_keypoints, links_dict, add_hat
    )

    # Adjust link direction if the link and keypoint directions do not match
    head_direction = get_head_direction(keypoints)
    if head_direction != links_dict["head"]["direction"]:
        links_dict["head"] = flip_link(links_dict["head"])
        links_dict["neck"] = flip_link(links_dict["neck"])
    links_dict = adjust_link_direction(links_dict, keypoints_dict)

    if add_hat:
        if head_direction != links_dict["hat"]["direction"]:
            links_dict["hat"] = flip_link(links_dict["hat"])

    h, w = background_img.shape[:2]

    canvas = background_img.copy()

    for link_name, link_data in links_dict.items():
        if link_name not in keypoints_dict:
            continue

        if (link_name == "head" and add_hat) or (link_name == "hat" and not add_hat):
            continue
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
    add_hat = False

    for persons_dict in keypoints_json["people"]:
        person_id = persons_dict["person_id"]
        print(f"Processing person with person_id {person_id}")
        background_img = generate_silhouette(
            persons_dict, links_dict, background_img, width_scale_factor, add_hat
        )

    cv2.imwrite(str(output_path), background_img)


if __name__ == "__main__":
    main()
