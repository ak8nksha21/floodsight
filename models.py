"""Model factory. Every model takes (pre, post) and returns 4 channels:
building, road, flooded_building, flooded_road."""
import torch
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


class SiameseUnet(nn.Module):
    """Siamese multi-task U-Net on top of smp's proven ResNet-34 U-Net encoder and decoder.
    use_pre=True : one shared encoder runs on the pre and the post image. At every level the
                   features are fused as  pre + g(concat[pre, post, |post - pre|])  with g a 1x1 conv
                   that starts at zero, so training begins as a normal pre-image U-Net and learns
                   to add the flood evidence from the change.
    use_pre=False: single-image multi-task baseline (post image only, same decoder and heads).
    Head 1 -> building, road.   Head 2 -> flooded building, flooded road.
    Dropout2d on the three deepest feature levels is what MC Dropout uses at inference."""

    def __init__(self, use_pre=True, weights="imagenet", dropout=0.1):
        super().__init__()
        base = smp.Unet("resnet34", encoder_weights=weights, classes=N_OUT)
        self.encoder, self.decoder = base.encoder, base.decoder
        self.use_pre = use_pre
        ch = list(self.encoder.out_channels)  # [3, 64, 64, 128, 256, 512]
        if use_pre:
            self.fuse = nn.ModuleList()
            for c in ch[1:]:
                g = nn.Conv2d(3 * c, c, 1)
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)
                self.fuse.append(g)
        self.drop = nn.ModuleList([nn.Dropout2d(dropout if i >= 2 else 0.0) for i in range(5)])
        self.head_infra = nn.Conv2d(16, 2, 3, padding=1)
        self.head_flood = nn.Conv2d(16, 2, 3, padding=1)

    def forward(self, pre, post):
        if self.use_pre:
            fa, fb = self.encoder(pre), self.encoder(post)  # same weights = Siamese
            feats = [fa[0]] + [a + g(torch.cat([a, b, (a - b).abs()], 1))
                               for g, a, b in zip(self.fuse, fa[1:], fb[1:])]
        else:
            feats = self.encoder(post)
        feats = [feats[0]] + [d(f) for d, f in zip(self.drop, feats[1:])]
        x = self.decoder(feats)
        return torch.cat([self.head_infra(x), self.head_flood(x)], dim=1)


def build_model(name, weights="imagenet", use="post", dropout=0.1):
    w = None if weights in (None, "none") else weights
    if name == "unet":
        return PostOnly(smp.Unet("resnet34", encoder_weights=w, classes=N_OUT), use)
    if name == "unetpp":
        return PostOnly(smp.UnetPlusPlus("resnet34", encoder_weights=w, classes=N_OUT), use)
    if name == "deeplab":
        return PostOnly(smp.DeepLabV3Plus("resnet34", encoder_weights=w, classes=N_OUT), use)
    if name == "single_mt":
        return SiameseUnet(use_pre=False, weights=w, dropout=dropout)
    if name == "siamese":
        return SiameseUnet(use_pre=True, weights=w, dropout=dropout)
    raise ValueError(f"unknown model: {name}")
