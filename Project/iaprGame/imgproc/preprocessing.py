
import cv2
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

class Imgproc():

    def __init__(self, img):
        self.img = img


    def preprocessing(self):
        """

        Arguments:

        Returns:

        """
        cropped_image = self.img[100: -300, :, :]

        img_green =cropped_image[:, :, 0]

        filter = cv2.medianBlur(img_green, 5)

        edges = cv2.Canny(filter, 0, 100)

        preprocessed = self.highpass(edges, 3)

        return cropped_image, preprocessed


    def highpass(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """

        Arguments:

        Returns:

        """
        return img - cv2.GaussianBlur(img, (0, 0), sigma)
