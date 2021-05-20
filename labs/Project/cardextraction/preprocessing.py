
import cv2

def preprocessing(img):
    """

    Arguments:

    Returns: 

    """
    cropped_image=img[100:-300,:,:]
    img_green=cropped_image[:,:,2]
    filter = cv2.medianBlur(img_green,5)  
    edges= cv2.Canny(filter,0,100)
    preprocessed=highpass(edges.copy(),3)
    return cropped_image, preprocessed


def highpass(img, sigma):
    """

    Arguments:

    Returns: 

    """
    return img - cv2.GaussianBlur(img, (0,0), sigma)


