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
        links_dict[link_name] = link_data

    torso_length = links_dict["torso"]["length"]

    for link_name, link_data in links_dict.items():
        links_dict[link_name]["relative_length"] = link_data["length"] / torso_length

    return links_dict


def parse_keypoints(keypoints, links_dict):
    keypoints_dict = {}
    for link_name, link_data in links_dict.items():
        dst_pt1 = keypoints[link_data["pt1_idx"]][:2]
        dst_pt2 = keypoints[link_data["pt2_idx"]][:2]
        length = np.linalg.norm(dst_pt2 - dst_pt1)
        keypoints_dict[link_name] = {}
        keypoints_dict[link_name]["length"] = length

    torso_length = keypoints_dict["torso"]["length"]

    for link_name, keypoint_link_data in keypoints_dict.items():
        keypoint_link_data["relative_length"] = (
            keypoint_link_data["length"] / torso_length
        )

    return keypoints_dict


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


def generate_silhouette(keypoints, links_dict, background_img, width_scale_factor=0.85):
    keypoint_dict = parse_keypoints(keypoints, links_dict)
    h, w = background_img.shape[:2]

    canvas = background_img.copy()

    for link_name, link_data in links_dict.items():
        src_pt1 = link_data["pt1"]
        src_pt2 = link_data["pt2"]
        link_img = link_data["image"]
        dst_pt1 = keypoints[link_data["pt1_idx"]][:2]
        dst_pt2 = keypoints[link_data["pt2_idx"]][:2]

        width_scale_coeff = (
            link_data["relative_length"]
            / keypoint_dict[link_name]["relative_length"]
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
        keypoints = persons_dict["pose_keypoints_2d"]
        keypoints_reshaped = np.array(keypoints).reshape((-1, 3))
        print(
            f"Person's with person_id {person_id} keypoints: \n{keypoints_reshaped}\n"
        )
        background_img = generate_silhouette(
            keypoints_reshaped, links_dict, background_img, width_scale_factor
        )

    cv2.imwrite(str(output_path), background_img)


if __name__ == "__main__":
    main()
