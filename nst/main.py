
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

# Create an instance of the VGGFeatures model, move it to the device, and set it to evaluation mode
model = VGGFeatures().to(device).eval()

# Pass the content
content_target, style_target = model(content_image)
_, style_feature = model(style_image)

# Define the content and style loss functions
content_loss = ContentLoss(content_target)
style_loss = [StyleLoss(style_feature[i]) for i in range(len(style_feature))]
