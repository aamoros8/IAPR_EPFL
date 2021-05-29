import numpy as np
import cv2
from typing import Tuple


def contour_finder(image: np.ndarray,
                   preprocessed: np.ndarray) -> Tuple(list, np.ndarray, bool):
    """
    Finds the 5 longest contours of the image which
    are the cards of the players as well as the dealer patch

    Arguments:
    image[np.ndarray]: originalimage for the contours to be drawn on
    preprocessed[np.ndarray]: preprocessed image

    Returns:
    contour [list]: 5 longtest contours of the iamge
    image_cont [np.ndarray]: Image with contours
    flag [bool]: flag for error detection

    """
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # draw all contours
    image_cont = cv2.drawContours(image.copy(), contours, -1, (255, 0, 0), 30)
    error = 0
    flag = False
    for cnt in contours[0:5]:
        if cv2.arcLength(cnt, True) < 1100:  # or cv2.arcLength(cnt,True)>3500:
            error += 1
            if error > 0:
                flag = True

    if flag:
        print('Contours not correctly identified,\
                please try something else bitch')

    return contours[0:5], image_cont, flag
