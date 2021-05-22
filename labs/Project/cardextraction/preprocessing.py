
import cv2
import numpy as np
from typing import Tuple


def preprocessing(img: np.ndarray) -> Tuple(np.ndarray, np.ndarray):
    """

    Arguments:

    Returns:

    """
    cropped_image = img[100: -300, :, :]
    img_green = cropped_image[:, :, 2]
    filter = cv2.medianBlur(img_green, 5)
    edges = cv2.Canny(filter, 0, 100)
    preprocessed = highpass(edges.copy(), 3)
    return cropped_image, preprocessed


def highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """

    Arguments:

    Returns:

    """
    return img - cv2.GaussianBlur(img, (0, 0), sigma)
