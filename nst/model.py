
import torch
import torch.nn as nn
import torchvision.models as models


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        # Define object vgg
        vgg = models.vgg19(pretrained = True).features

        # freeze parameters
        for param in vgg.parameters():
            param.requires_grad_(False)

        # Assign vgg to self.vgg
        self.vgg = vgg

        # Get the content and style layers
        self.content_layers = '21'
        self.style_layers = ['0', '5', '10', '19', '28']
    

    def forward(self, x):
        content = None
        style = []

        for name, layer in self.vgg._modules.items():
            x = layer(x)

            # Reach the content layers, save these features
            if name == self.content_layers:
                content = x

            # Reach the style layers, save these features
            if name in self.style_layers:
                style.append(x)


        return content, style
