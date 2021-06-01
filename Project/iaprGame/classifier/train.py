import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim 
from torchvision import datasets
import numpy as np

def train_model(model:nn.Module, train_input:torch.tensor, train_target:torch.tensor,
               test_input:torch.tensor ,test_target:torch.tensor , mini_batch_size: int,
            nb_epochs: int = 10,verbose:int =0):

    """
    Training model function, plots the final train and test error 

    Attributes: 
    criterion: MSE Loss
    Optimizer: SGD 

    Parameters: 
    model [nn.Module]: Pytorch module wished to be trained
    train_input [torch.tensor]: input images with dimension (n,1,28,28) with n the number of images
    train_target [torch.tesnor]: input labels with dimension(n,1)
    mini_batch_size [int]: Batch Size
    nb_epochs [int]: amount of epochs,  20 by default
    verbose[int]: If >0, displays the train and test error for each epoch. 

    Returns: 
    train_error [numpy.ndarray]: array with train error pourcentage with dimension (1,nb_epoch)
    test_error [numpy.ndarray]:array with train error pourcentage with dimension (1,nb_epoch)
    """

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1, momentum=0.9)

    train_error=[]
    test_error=[]
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))

            if type(criterion)==torch.nn.modules.loss.MSELoss:
                 targets = train_target.narrow(0, b, mini_batch_size)
            elif type(criterion)==torch.nn.modules.loss.CrossEntropyLoss:
                _, targets = train_target.narrow(0, b, mini_batch_size).max(dim=1)

            loss = criterion(output,targets)
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        nb_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
        nb_test_errors =  compute_nb_errors(model, test_input, test_target, mini_batch_size)

        train_error=np.append(train_error,(100 * nb_train_errors) / train_input.size(0))
        test_error=np.append(test_error,(100 * nb_test_errors) / test_input.size(0))


        if verbose>0:
            if (e %10 == 0):
                print(e,' Train error: {:0.2f}%  Test error: {:0.2f}% '.format((100 * nb_train_errors) / train_input.size(0),
                                                                        (100 * nb_test_errors) / test_input.size(0))) 
            if verbose>1:
                print(e,' Train error: {:0.2f}%  Test error: {:0.2f}% '.format((100 * nb_train_errors) / train_input.size(0),
                                                                        (100 * nb_test_errors) / test_input.size(0))) 

        if e==nb_epochs-1:
            print('Final Train error: {:0.2f}%  Test error: {:0.2f}% '.format((100 * nb_train_errors) / train_input.size(0),
                                                                        (100 * nb_test_errors) / test_input.size(0)))
    return train_error,test_error
        
           




def compute_nb_errors(model:nn.Module, input:torch.tensor, target:torch.tensor, mini_batch_size:int)->int:
    """
    Computes the absolute number of errors of a model with respect to an input and a target 

    Parameters:
    model [nn.Module]: Model to be used
    input[torch.tensor]: input images with dimension (n,1,28,28) with n the number of images
    target [torch.tesnor]: input labels with dimension(n,1)
    mini_batch_size [int]: Batch Size

    Returns:
    nb_errors[int]: Absolute number of errors where the target differs from the predicted value

    """
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k, predicted_classes[k]] <= 0.5:
                nb_errors = nb_errors + 1

    return nb_errors