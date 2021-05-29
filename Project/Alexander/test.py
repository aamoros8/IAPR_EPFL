import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from functional import *
game,i=4,8 #3,1  4,8 5,3. 5,7 5,8 6,1 6,8
img=cv2.imread("train_games/game{0}/{1}.jpg".format(game,i))

img,contours, image_cont=contour_finder(img)
plt.imshow(image_cont)
plt.show()
find_players(contours,img,game,i)
# show the image with the drawn contours
