from imgaug import augmenters as iaa

def augment_data_imgaug(data, labels, nb_op):
    """
    Augment operators dataset to have more images + be resistant to different kind of transfo 
    """
    
    data_aug, labels_aug = [], []

    #define the possible transformations 
    seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.LinearContrast((0.75, 1.5)), # strengthen or weaken the contrast in each image
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # Add gaussian noise.
    iaa.Multiply((0.8, 1.2), per_channel=0.2), # Make some images brighter and some darker.
    iaa.Affine(   # Apply affine transformations to each image.
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-90, 90),
        shear=(-8, 8) 
    )], random_order=True) # apply augmenters in random order

    #nb of sample per class (per operator)
    sample_per_class=1000

    for batch_idx in range(int(sample_per_class)): 
        for i in range(nb_op): 
            data_aug.append(seq.augment_images(chooseImage(data, labels, i))) #add one image
            labels_aug.append(i) #add the corresponding label

    return data_aug, labels_aug