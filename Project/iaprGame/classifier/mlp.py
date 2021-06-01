import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Multi Layer Perceptron Module 

        Arguments: 
        nb_hidden [int]: Dimension of the hidden layer 

        Attributes: 
        fc1: First fully connected linear layer (28*28) -->nb_hidden 
        fc2: Second fully connected linear layer (nb_hidden) --> 10 

        """
        # First fully connected linear layer (784)->(100)
        self.fc1 = nn.Linear(28*28, 100) 
        self.fc2 = nn.Linear(100, 100) 
        self.fc3 = nn.Linear(100, 50) 
        self.fc4 = nn.Linear(50, 3) 

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 28*28)))
        x= F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x= self.fc4(x)
        return x
