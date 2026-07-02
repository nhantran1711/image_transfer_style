
import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    # Init Content Loss object
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    # Forward pass of the Content Loss object
    def forward(self, input):
        return torch.mean((input - self.target) ** 2)


def gram_matrix(input):
    # Get the batch size, channels, height and width of the input tensor
    batch_size, channels, height, width = input.size()

    # Calculate the Gram matrix by reshaping the input tensor and performing a matrix multiplication
    features = input.view(batch_size * channels, height * width)

    # Calculate the Gram matrix by performing a matrix multiplication of the features with its transpose
    gram = torch.mm(features, features.t())
    return gram / (channels * height * width)

class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        # Calculate the Gram matrix of the target features and detach it from the computation graph
        self.target = gram_matrix(target).detach()
    
    def forward(self, input):
        
        # Calculate the Gram matrix of the input features
        gram = gram_matrix(input)
        return torch.mean((gram - self.target) ** 2)