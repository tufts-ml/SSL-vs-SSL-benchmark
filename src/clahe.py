import cv2 
import numpy as np
from PIL import Image


def apply_clahe(image):
    np_image = np.array(image)  
    if np_image.ndim == 2:  
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        np_image = clahe.apply(np_image)
    else:  
        lab_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab_image = cv2.merge((l, a, b))
        np_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return Image.fromarray(np_image)