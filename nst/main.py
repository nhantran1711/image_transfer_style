
import torch
import torch.optim as optim

# From 3 files
from model import VGGFeatures
from utils import load_image 
from losses import ContentLoss, StyleLoss

# Define the main function for neural style transfer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the content and style images and move them to the device (CPU or GPU)
content_image = load_image("images/content.jpg").to(device)
style_image = load_image("images/style.jpg").to(device)

