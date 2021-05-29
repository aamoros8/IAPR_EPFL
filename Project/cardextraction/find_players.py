
import cv2
import numpy as np


def find_players(contours: list, img: np.ndarray) -> list:

    """

    Arguments:

    Returns: 

    """

    classes=[]
    cards=[]

    for cnt in contours[0:5]:
        x, y, w, h = cv2.boundingRect(cnt)
        classes=img[y:y + h, x:x + w]
        cards.append(card( x, y, w, h,cnt))

    centers=cards[0].center()
    for crd in cards[1:-1]: 
        centers=np.vstack((centers,crd.center()))

    idx_y_min=np.argmin(centers[:,1])
    idx_y_max=np.argmax(centers[:,1])
    idx_x_min=np.argmin(centers[:,0])
    idx_x_max=np.argmax(centers[:,0])

    cards[idx_y_max].player='Player 1'
    cards[idx_x_max].player='Player 2'
    cards[idx_y_min].player='Player 3'
    cards[idx_x_min].player='Player 4'
    cards[-1].player='Dealer'

    return cards

      

class card:
    """

    Arguments:

    Returns: 

    """

    def __init__(self, x, y, w, h, cnt):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.player=None
        self.contour=cnt
    
    def center(self)-> np.array:

        center=np.array([self.x + self.w / 2, self.y + self.h / 2])
        return center

    def set_player(self, player: str):
        self.player = player
  