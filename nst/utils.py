

# Preprocessing functions for neural style transfer.

from tkinter import Image
import torchvision.transforms as transforms

def load_image(image_path, max_size = 512):

    # Get the image from path and convert it to RGB
    image = Image.open(image_path).convert('RGB')

    # Resize the image if it is larger than max_size
    size = max(image.size)

    if size > max_size:
        scale = max_size / size
        new_size = tuple([int(dim * scale) for dim in image.size])
    else:
        new_size = image.size
    
    # Transform the image to a tensor
    transformed_image = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor(),
    ])
    
    # Add a batch dimension to the image tensor
    image = transformed_image(image).unsqueeze(0)

    return image