"""Model factory. Every model takes (pre, post) and returns 4 channels:
building, road, flooded_building, flooded_road."""
import torch.nn as nn
import segmentation_models_pytorch as smp

N_OUT = 4


class PostOnly(nn.Module):
    """Single-image model. `use` picks the image it sees: "post" (default) or "pre"."""

    def __init__(self, net, use="post"):
        super().__init__()
        self.net, self.use = net, use

    def forward(self, pre, post):
        return self.net(post if self.use == "post" else pre)


def build_model(name, weights="imagenet", use="post"):
    w = None if weights in (None, "none") else weights
    if name == "unet":
        return PostOnly(smp.Unet("resnet34", encoder_weights=w, classes=N_OUT), use)
    if name == "unetpp":
        return PostOnly(smp.UnetPlusPlus("resnet34", encoder_weights=w, classes=N_OUT), use)
    if name == "deeplab":
        return PostOnly(smp.DeepLabV3Plus("resnet34", encoder_weights=w, classes=N_OUT), use)
    raise ValueError(f"unknown model: {name}")
