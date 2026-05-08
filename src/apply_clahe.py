import skimage as ski
import skimage.io
import numpy as np
from PIL import Image

def apply_clahe(image):
    """Applies Contrast Limited Adaptive Histogram Equalization to an image

    Args:
        image (PIL Image): image to apply CLAHE to
    
    Returns:
        PIL Image representing the transformed input
    """
    img_array = np.array(image, dtype=np.uint8)
    clahe_image = ski.exposure.equalize_adapthist(img_array)
    return Image.fromarray(np.uint8(clahe_image*255), mode="RGB")