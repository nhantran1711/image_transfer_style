
import torch
import torch.optim as optim
from torchvision.utils import save_image

# From 3 files
from model import VGGFeatures
from utils import load_image 
from losses import ContentLoss, StyleLoss


# VARIABLES
STYLE_WEIGHT = 1e6
CONTENT_WEIGHT = 1

running = [0]


# Define the main function for neural style transfer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the content and style images and move them to the device (CPU or GPU)
content_image = load_image("images/content.jpg").to(device)
style_image = load_image("images/style.jpg").to(device)

# Create an instance of the VGGFeatures model, move it to the device, and set it to evaluation mode
model = VGGFeatures().to(device).eval()

# Pass the content and style images through the model
content_target, style_target = model(content_image)
_, style_feature = model(style_image)

# Define the content and style loss functions
content_loss = ContentLoss(content_target)
style_loss = [StyleLoss(style_feature[i]) for i in range(len(style_feature))]

# Init generated image as a clone of the content image and set requires_grad to True for optimization
generated_image = content_image.clone().requires_grad_(True)

# Define the optimizer for the generated image using Adam optimizer with a learning rate of 0.01
optimizer = optim.Adam([generated_image], lr = 0.01)


# Main loop 
while running[0] <= 300:

    def closure():
        optimizer.zero_grad()  # Clear the gradients of the optimizer
    
        content_feat, style_feat = model(generated_image)  # Pass the generated image through the model

        content_loss_value = content_loss(content_feat)  # Calculate the content loss

        # Calculate the style loss 
        style_loss_value = 0 

        for i in range(len(style_feat)):
            style_loss_value += style_loss[i](style_feat[i])  # Accumulate the style loss for each layer
        
        # Calculate the total loss as a weighted sum of content and style losses
        total_loss = CONTENT_WEIGHT * content_loss_value + STYLE_WEIGHT * style_loss_value
          
        running[0] += 1

        if running[0] % 50 == 0:
            print(f"Iteration {running[0]}: Total Loss: {total_loss.item()}") 

        return total_loss  
    
    # Perform an optimization step using the closure function
    optimizer.step(closure)  


# Save the final generated image to a file 
save_image(generated_image, "images/generated_image.jpg")