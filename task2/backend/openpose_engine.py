import subprocess
from pathlib import Path


class OpenPoseEngine:
    def __init__(
        self,
        openpose_root_dir: Path,
        net_resolution: str = "-1x368",
        hand_net_resolution: str = "368x368",
    ):
        self.openpose_root_dir = openpose_root_dir
        self.executable_path = openpose_root_dir / "bin" / "OpenPoseDemo.exe"
        self.net_resolution = net_resolution
        self.hand_net_resolution = hand_net_resolution

    def extract_pose(self, image_dir: Path, output_dir: Path) -> None:
        """
        Extracts human poses from images located in the input directory using OpenPose.
        Saves the results as JSON files in the output directory.
        """
        try:
            command = [
                str(self.executable_path),
                "--image_dir",
                str(image_dir),
                "--write_json",
                str(output_dir),
                "--write_images",
                str(output_dir),
                "--display",
                "0",
                "--render_pose",
                "1",
                "--hand",
                "--net_resolution",
                self.net_resolution,
                "--hand_net_resolution",
                self.hand_net_resolution,
            ]
            subprocess.run(command, check=True, cwd=self.openpose_root_dir)
        except Exception as e:
            print(f"Error in OpenPose detection: {e}")
            raise e
