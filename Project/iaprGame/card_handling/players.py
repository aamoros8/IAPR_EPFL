
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import inspect
from ..card import card


class Players():

    def __init__(self) -> None:
        pass
        
    def contour_finder(self, image: np.ndarray,
                    preprocessed: np.ndarray):
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
            print('')

        return contours[0:5], image_cont, flag
    

    def find_players(self, contours: list, img: np.ndarray) -> list:

        """

        Arguments:

        Returns: 

        """

        #classes=[]
        cards=[]

        for cnt in contours[0:5]:
            x, y, w,h = cv2.boundingRect(cnt)
            rect=cv2.minAreaRect(cnt)
            cards.append(card( x, y, w, h,cnt,rect))

        centers=cards[0].center()
        for crd in cards[1:-1]: 
            centers=np.vstack((centers,crd.center()))

        idx_y_min=np.argmin(centers[:,1])
        idx_y_max=np.argmax(centers[:,1])
        idx_x_min=np.argmin(centers[:,0])
        idx_x_max=np.argmax(centers[:,0])

        idx=np.array([idx_y_max, idx_x_max, idx_y_min, idx_x_min])

        if centers[idx_y_max,1] < img.shape[1]/2:
            print('Player 1 not correctly detected')

        elif centers[idx_x_max,0] < img.shape[1]/2:
            print('Player 2 not correctly detected')

        elif centers[idx_y_min,1] > img.shape[0]/2:
            print('Player 3 not correctly detected')

        elif centers[idx_x_min,0] > img.shape[0]/2:
            print('Player 4 not correctly detected')
        
        elif np.any(idx==4):
            print('Problem Player classified to dealer')

        elif not np.unique(idx).shape[0]==4:
            print('Problem, multiple players assigned to same card')

        cards[idx_y_max].player='Player 1'
        cards[idx_x_max].player='Player 2'
        cards[idx_y_min].player='Player 3'
        cards[idx_x_min].player='Player 4'
        cards[-1].player='Dealer'

        return cards


    def draw_players(self, img,cards,game,pic):
        """

        Arguments:

        Returns: 

        """
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        image=img.copy()

        p=0
        for crd in cards: 
            if crd.player=='Player 1':
                loc_text=(crd.center()-[crd.w/2,500]).astype(int)
            elif crd.player=='Player 2' or crd.player=='Player 3' or crd.player=='Player 4':
                loc_text=(crd.center()+[-crd.w/2,crd.h]).astype(int)
            elif crd.player=='Dealer':
                loc_text=(crd.center()-[crd.w,400]).astype(int)
            
            cv2.putText(image,crd.player,tuple(loc_text), font, 5,(0,0,0),20)
            p+=1
            cv2.rectangle(image, (crd.x, crd.y),(crd.x+crd.w,crd.y+crd.h), (255,0,0), 50)

        plt.imshow(image)
        plt.title('Game{0} Pic {1}'.format(game,pic))
        plt.show()


        

      

