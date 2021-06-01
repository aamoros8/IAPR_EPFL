import numpy as np

class card:
    """

    Arguments:

    Returns: 

    """

    def __init__(self, x, y, w, h, cnt,rect):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.rect=rect
        self.player=None
        self.contour=cnt
        self.value=None
        self.family=None 
    
    def center(self)-> np.array:

        center=np.array([self.x + self.w / 2, self.y + self.h / 2])
        return center

    def set_player(self, player: str):
        self.player = player