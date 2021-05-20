import cv2
import matplotlib.pyplot as plt

def draw_players(img,cards,game,pic):
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


    