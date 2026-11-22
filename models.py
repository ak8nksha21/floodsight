"""Model factory. Every model takes (pre, post) and returns 4 channels:
building, road, flooded_building, flooded_road. Single-image models ignore `pre`."""
import torch.nn as nn
import segmentation_models_pytorch as smp

N_OUT = 4


class PostOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pre, post):
        return self.net(post)


def build_model(name, weights="imagenet"):
    w = None if weights in (None, "none") else weights
    if name == "unet":
        return PostOnly(smp.Unet("resnet34", encoder_weights=w, classes=N_OUT))
    if name == "unetpp":
        return PostOnly(smp.UnetPlusPlus("resnet34", encoder_weights=w, classes=N_OUT))
    if name == "deeplab":
        return PostOnly(smp.DeepLabV3Plus("resnet34", encoder_weights=w, classes=N_OUT))
    raise ValueError(f"unknown model: {name}")
