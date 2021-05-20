import numpy as np
import cv2
import matplotlib.pyplot as plt


def contour_finder(image,preprocessed):
    """

    Arguments:

    Returns: 

    """
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    contours = sorted(contours, key=cv2.contourArea,reverse=True) 

    # draw all contours
    image_cont = cv2.drawContours(image.copy(), contours, -1, (255, 0, 0), 30)
    error=0
    flag=False
    for cnt in contours[0:5]:
        if cv2.arcLength(cnt,True)<1100: #or cv2.arcLength(cnt,True)>3500:
            error+=1
            if error>0:
                flag=True

    if flag: print('Contours not correctly identified, please try something else bitch')

    return contours[0:5], image_cont, flag
