import numpy as np
import cv2
import matplotlib.pyplot as plt


def contour_finder(img):
    img=normalize(img[100:-300,:,:])  
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray=img[:,:,2]
    ksize=3
    ##sigma=0.3*((ksize-1)*0.5 - 1) + 0.8
    #filter=wiener_filter(img_gray,kernel=cv2.getGaussianKernel(ksize,sigma),K=10)
    filter = cv2.medianBlur(img_gray,5)
    #filter= cv2.bilateralFilter(img_gray,9,75,75)
    #plt.imshow(filter,cmap='gray')
    #plt.show()
    edges= cv2.Canny(filter,100,200)
    #plt.imshow(edges,cmap='gray')
    #plt.show()
    edges=highpass(edges.copy(),3)
    plt.imshow(edges,cmap='gray')
    plt.show()
    
    # find the contours from the thresholded image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    contours = sorted(contours, key=cv2.contourArea,reverse=True) 

    # draw all contours
    image_cont = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 30)
    

    return img,contours, image_cont

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma)

def find_players(contours,img,game,pic):
    idx=0
    classes=[]
    cards=[]
    for cnt in contours[0:5]:
        idx+=1
        x, y, w, h = cv2.boundingRect(cnt)
        classes=img[y:y + h, x:x + w]
        cards.append(card( x, y, w, h))
        #plt.imshow(classes)
        #plt.show()
        #cv2.imwrite(str(idx) + '.png', classes)

    centers=cards[0].center()
    for crd in cards[1:-1]: 
        centers=np.vstack((centers,crd.center()))

    idx_y_min=np.argmin(centers[:,1])
    idx_y_max=np.argmax(centers[:,1])
    idx_x_min=np.argmin(centers[:,0])
    idx_x_max=np.argmax(centers[:,0])

    cards[idx_y_max].player='Player 1'
    cards[idx_y_max].contour=contours[idx_y_max]
    cards[idx_x_max].player='Player 2'
    cards[idx_x_max].contour=contours[idx_x_max]
    cards[idx_y_min].player='Player 3'
    cards[idx_y_min].contour=contours[idx_y_min]
    cards[idx_x_min].player='Player 4'
    cards[idx_x_min].contour=contours[idx_x_min]
    cards[-1].player='Dealer'
    cards[-1].contour=contours[5]
    #print('Player 1 has index', idx_y_max)
    #print('Player 2 has index', idx_x_max)
    #print('Player 3 has index', idx_y_min)
    #print('Player 4 has index', idx_x_min)

    font = cv2.FONT_HERSHEY_SIMPLEX
    image=img.copy()
    Players=['Player 1', 'Player 2', 'Player 3','Player 4', 'Dealer']
    IDX=[idx_y_max,idx_x_max, idx_y_min, idx_x_min, -1]

    p=0
    for i in IDX: 
        if p==0:
            loc_text=(cards[i].center()-[cards[i].w/2,500]).astype(int)
        elif p==1 or p==3 or p==2:
            loc_text=(cards[i].center()+[-cards[i].w/2,cards[i].h]).astype(int)
        elif p==4:
            loc_text=(cards[i].center()-[cards[i].w,400]).astype(int)
        
        cv2.putText(image,Players[p],tuple(loc_text), font, 5,(0,0,0),20)
        p+=1
        cv2.rectangle(image, (cards[i].x, cards[i].y),(cards[i].x+cards[i].w,cards[i].y+cards[i].h), (255,0,0), 50)

    plt.imshow(image)
    plt.title('Game{0} Pic {1}'.format(game,pic))
    plt.show()

    return cards


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr.astype('uint8')

def wiener_filter(img,kernel,K):
    kernel/=np.sum(kernel)
    dummy=np.copy(img)
    dummy=np.fft.fft2(dummy)
    kernel=np.fft.fft2(kernel,s=img.shape)
    kernel=np.conj(kernel)/(np.abs(kernel)**2+K)
    dummy=dummy*kernel
    dummy=np.abs(np.fft.ifft2(dummy))
    return dummy.astype('uint8')
    
class card:
    def __init__(self,x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.player=None
        self.contour=None
        self.main_colour = None
    
    def center(self)-> np.array:
        center=np.array([self.x + self.w/2,self.y + self.h/2])
        return center

    def set_player(self,player:str):
        self.player=player
        
    
