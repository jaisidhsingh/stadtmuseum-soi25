import numpy as np
from rembg import remove, new_session
from PIL import Image
import cv2

class SegmentationEngine:
    def __init__(self):
        # Initialize session once
        self.session = new_session("u2net_human_seg")

    def extract_silhouette(self, image_path: str) -> Image.Image:
        """
        Extracts the person from the image using rembg U2NET Human Seg.
        Returns a PIL Image (RGBA) with the background removed.
        """
        try:
            input_image = Image.open(image_path).convert("RGB")
            # Convert to numpy for rembg if needed, or pass PIL directly. 
            # rembg supports PIL images directly.
            
            output_image = remove(input_image, session=self.session)

            # Create black silhouette from alpha channel
            if output_image.mode == 'RGBA':
                r, g, b, a = output_image.split()
                # Create black image of same size
                black_bg = Image.new('RGB', output_image.size, (0, 0, 0))
                # Put alpha channel back
                output_image = black_bg.convert('RGBA')
                output_image.putalpha(a)
            
            return output_image
        except Exception as e:
            print(f"Error in segmentation: {e}")
            raise e
