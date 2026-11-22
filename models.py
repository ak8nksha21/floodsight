"""Model factory. Every model takes (pre, post) and returns 4 channels:
building, road, flooded_building, flooded_road."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

N_OUT = 4


class PostOnly(nn.Module):
    """Single-image model. `use` picks the image it sees: "post" (default) or "pre"."""

    def __init__(self, net, use="post"):
        super().__init__()
        self.net, self.use = net, use

    def forward(self, pre, post):
        return self.net(post if self.use == "post" else pre)


def conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    """Upsample x2, join the skip connection, two convs, then Dropout2d (used later for MC Dropout)."""

    def __init__(self, cin, skip, cout, p):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(cin + skip, cout), conv_bn_relu(cout, cout))
        self.drop = nn.Dropout2d(p)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.drop(self.body(x))


class SiameseMultiTask(nn.Module):
    """Shared ResNet-34 encoder -> fusion at every level -> U-Net decoder -> two 1x1 heads.
    use_pre=True : Siamese model. Fusion = concat[post, pre, |post - pre|] -> 1x1 conv.
    use_pre=False: same network, pre-flood image ignored (single-image multi-task baseline).
    Head 1 -> building, road.   Head 2 -> flooded building, flooded road."""

    def __init__(self, use_pre=True, weights="imagenet", dropout=0.2):
        super().__init__()
        self.use_pre = use_pre
        self.encoder = smp.encoders.get_encoder("resnet34", in_channels=3, depth=5, weights=weights)
        ch = list(self.encoder.out_channels)  # [3, 64, 64, 128, 256, 512]
        n_in = 3 if use_pre else 1
        self.fuse = nn.ModuleList([conv_bn_relu(c * n_in, c, k=1) for c in ch[1:]])
        dec = [256, 128, 64, 32, 16]
        self.blocks = nn.ModuleList([
            DecoderBlock(ch[5], ch[4], dec[0], dropout),
            DecoderBlock(dec[0], ch[3], dec[1], dropout),
            DecoderBlock(dec[1], ch[2], dec[2], dropout),
            DecoderBlock(dec[2], ch[1], dec[3], dropout),
            DecoderBlock(dec[3], 0, dec[4], dropout),
        ])
        self.head_infra = nn.Conv2d(dec[4], 2, 1)
        self.head_flood = nn.Conv2d(dec[4], 2, 1)

    def forward(self, pre, post):
        fp = self.encoder(post)[1:]
        if self.use_pre:
            fb = self.encoder(pre)[1:]  # same weights = Siamese
            feats = [f(torch.cat([a, b, (a - b).abs()], 1)) for f, a, b in zip(self.fuse, fp, fb)]
        else:
            feats = [f(a) for f, a in zip(self.fuse, fp)]
        x = self.blocks[0](feats[4], feats[3])
        x = self.blocks[1](x, feats[2])
        x = self.blocks[2](x, feats[1])
        x = self.blocks[3](x, feats[0])
        x = self.blocks[4](x, None)
        return torch.cat([self.head_infra(x), self.head_flood(x)], dim=1)


def build_model(name, weights="imagenet", use="post", dropout=0.2):
    w = None if weights in (None, "none") else weights
    if name == "unet":
        return PostOnly(smp.Unet("resnet34", encoder_weights=w, classes=N_OUT), use)
    if name == "unetpp":
        return PostOnly(smp.UnetPlusPlus("resnet34", encoder_weights=w, classes=N_OUT), use)
    if name == "deeplab":
        return PostOnly(smp.DeepLabV3Plus("resnet34", encoder_weights=w, classes=N_OUT), use)
    if name == "single_mt":
        return SiameseMultiTask(use_pre=False, weights=w, dropout=dropout)
    if name == "siamese":
        return SiameseMultiTask(use_pre=True, weights=w, dropout=dropout)
    raise ValueError(f"unknown model: {name}")
