import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from timm.models.layers import DropPath

from .common import SCDNet


def add_drop_path_to_resnet(model, drop_path_rate=0.3):
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    total_blocks = sum(len(layer) for layer in layers)
    block_idx = 0

    for layer in layers:
        for block in layer:
            drop_rate = drop_path_rate * block_idx / max(total_blocks - 1, 1)
            block_idx += 1
            block.drop_path = DropPath(drop_rate) if drop_rate > 0 else nn.Identity()

            old_forward = block.forward

            def new_forward(self, x, old_forward=old_forward):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if hasattr(self, "conv3"):
                    out = self.relu(out)
                    out = self.conv3(out)
                    out = self.bn3(out)

                out = self.drop_path(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)
                return out

            block.forward = new_forward.__get__(block, block.__class__)

    return model


class ResNet50Encoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        f1 = self.base.layer1(x)
        f2 = self.base.layer2(f1)
        f3 = self.base.layer3(f2)
        f4 = self.base.layer4(f3)
        return [f1, f2, f3, f4]


def build_model(num_classes, input_size, output_size, drop_rate, pretrained_path=None, freeze_backbone=False):
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_path is None else None
    base = resnet50(weights=weights)
    base = add_drop_path_to_resnet(base, drop_path_rate=drop_rate)

    if pretrained_path is not None:
        import torch
        state_dict = torch.load(pretrained_path, map_location="cpu")
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        base.load_state_dict(state_dict, strict=False)
        print(f"ResNet50 checkpoint loaded: {pretrained_path}")

    if freeze_backbone:
        for p in base.parameters():
            p.requires_grad = False
        print("ResNet50 backbone is frozen.")

    encoder = ResNet50Encoder(base)
    return SCDNet(
        encoder=encoder,
        in_channel_list=[256, 512, 1024, 2048],
        num_classes=num_classes,
        output_size=output_size,
        drop_rate=drop_rate,
        out_channels=128,
        use_refinement_block=True,
    )