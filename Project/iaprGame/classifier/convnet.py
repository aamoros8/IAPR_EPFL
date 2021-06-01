
import torch
from torch import nn



class ConvNet(nn.Module):
    """
    Convolutional Network Module

    Attributes:
        conv1 (nn.Conv2d)     : fist convolutional layer
        conv2 (nn.Conv2d)     : second convolutional layer
        fc1 (nn.Linear)       : first fully connected layer
        fc2 (nn.Linear)       : second fully connected layer
        fc3 (nn.Linear)       : third fully connected layer
        fc4 (nn.Linear)       : last fully connected layer
        drop (nn.Dropout)     : dropout function
        drop2d (nn.Dropout)   : dropout function that drop entires channels
        pool (nn.MaxPool2d)   : maxpool function
        relu (nn.Relu)        : relu activation function
        sigmoid (nn.Sigmoid)  : sigmoid activation function
    """
    
    def __init__(self) -> None:
        """Initialize Convolutional Neural Network"""
        super().__init__()
    
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)   # 28x28x1  =>  26 x 26 x 20
        
        # fully connected layers
        self.fc1 = nn.Linear(13*13*20, 100)
        self.fc2 = nn.Linear(100, 13)
    
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass function

        Args:
            x (torch.tensor): Input tensors of dimensions [B, 2, 14, 14] with B being batch size

        Returns:
            torch.tensor: Predicted probability of size [1]
        """

        x = self.conv1(x)
        x = self.relu(self.pool(x))
        x = self.fc1(x.flatten(start_dim=1))
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.softmax(x)

        return x
    
    def __str__(self) -> str:
        """Representation"""
        return "Convolutional Neural Network"