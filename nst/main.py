
import argparse
import os

import torch
import torch.optim as optim
from torchvision.utils import save_image

# From 3 files
from model import VGGFeatures
from utils import load_image
from losses import ContentLoss, StyleLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "..", "images")


def parse_args():
    parser = argparse.ArgumentParser(description="Neural style transfer")
    parser.add_argument("--content", default=os.path.join(IMAGES_DIR, "content.jpg"),
                         help="Path to the content image")
    parser.add_argument("--style", default=os.path.join(IMAGES_DIR, "style.jpg"),
                         help="Path to the style image")
    parser.add_argument("--output", default=os.path.join(IMAGES_DIR, "generated_image.jpg"),
                         help="Path to save the generated image")
    parser.add_argument("--steps", type=int, default=300, help="Number of optimization steps")
    parser.add_argument("--style-weight", type=float, default=1e6, help="Weight for style loss")
    parser.add_argument("--content-weight", type=float, default=1, help="Weight for content loss")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the optimizer")
    parser.add_argument("--max-size", type=int, default=512, help="Max image dimension")
    parser.add_argument("--print-every", type=int, default=50, help="Print loss every N steps")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the content and style images and move them to the device (CPU or GPU)
    content_image = load_image(args.content, max_size=args.max_size).to(device)
    style_image = load_image(args.style, max_size=args.max_size).to(device)

    # Create an instance of the VGGFeatures model, move it to the device, and set it to evaluation mode
    model = VGGFeatures().to(device).eval()

    # Pass the content and style images through the model
    content_target, _ = model(content_image)
    _, style_feature = model(style_image)

    # Define the content and style loss functions
    content_loss = ContentLoss(content_target)
    style_loss = [StyleLoss(style_feature[i]) for i in range(len(style_feature))]

    # Init generated image as a clone of the content image and set requires_grad to True for optimization
    generated_image = content_image.clone().requires_grad_(True)

    # Define the optimizer for the generated image using Adam optimizer
    optimizer = optim.Adam([generated_image], lr=args.lr)

    running = [0]

    # Main loop
    while running[0] <= args.steps:

        def closure():
            optimizer.zero_grad()  # Clear the gradients of the optimizer

            content_feat, style_feat = model(generated_image)  # Pass the generated image through the model

            content_loss_value = content_loss(content_feat)  # Calculate the content loss

            # Calculate the style loss
            style_loss_value = 0

            for i in range(len(style_feat)):
                style_loss_value += style_loss[i](style_feat[i])  # Accumulate the style loss for each layer

            # Calculate the total loss as a weighted sum of content and style losses
            total_loss = args.content_weight * content_loss_value + args.style_weight * style_loss_value
            total_loss.backward()

            running[0] += 1

            if running[0] % args.print_every == 0:
                print(f"Iteration {running[0]}: Total Loss: {total_loss.item()}")

            return total_loss

        # Perform an optimization step using the closure function
        optimizer.step(closure)

    # Save the final generated image to a file
    save_image(generated_image, args.output)
    print(f"Saved generated image to {args.output}")


if __name__ == "__main__":
    main()
