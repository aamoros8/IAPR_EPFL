from imgaug import augmenters as iaa
import sys
import cv2
import gzip
import numpy as np
import torch
import torch.nn as nn

def data_aug(sample_per_class:int = 10, mode:str = 'full'):
    if mode == 'full':
        label=[10,11,12]
    if mode == 'reduced':
        label = [0 ,1,2]

    card=['king', 'queen', 'sword']

    seq = iaa.Sequential([
        #iaa.Fliplr(0.5), # horizontal flips
        #iaa.Crop(percent=(0, 0.1)), # random crops
        #iaa.LinearContrast((0.75, 1.5)), # strengthen or weaken the contrast in each image
        #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # Add gaussian noise.
        #iaa.Multiply((0.8, 1.2), per_channel=0.2), # Make some images brighter and some darker.
        iaa.Affine(   # Apply affine transformations to each image.
            scale={"x": (0.1, 1), "y": (0.1, 1)},
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-90, 90),
            #shear=(-8, 8) 
        )
        ], random_order=True) # apply augmenters in random order

    #nb of sample per class (per operator)

    data_aug, labels_aug = [], []

    for i in range(len(label)):
        path = 'test/{0}.jpg'.format(card[i]) 
        img = cv2.imread((path))
        if img is None:
            sys.exit('Image not found')

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_gray = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)

        for j in range(sample_per_class):
            data_aug.append(seq(images=img_gray)) #add one image
            labels_aug.append(label[i]) #add the corresponding label

    return labels_aug, data_aug


def extract_data(filename, image_shape, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def convert_to_one_hot_labels(input:torch.tensor, target:torch.tensor)->torch.tensor:
    """
    Conversion to one hot labels

    Parameters: 
    input[torch.tensor]: input images with dimension (n,1,28,28) with n the number of images
    target [torch.tesnor]: input labels with dimension(n,1)

    Returns:
    tmp [torch.tensor]: One hot label tensor with dimension (n,c) with n the number of data points and c the number of classes.    
    """

    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp
