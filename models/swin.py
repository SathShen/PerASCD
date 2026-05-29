import torch
import timm
from safetensors.torch import load_file

from .common import SCDNet


def build_encoder(drop_rate, pretrained_path=None, freeze_backbone=False):
    encoder = timm.create_model(
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        pretrained=False,
        features_only=True,
        img_size=512,
        drop_path_rate=drop_rate,
    )

    ckpt_path = pretrained_path
    if ckpt_path is not None:
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        converted_sd = {k.replace("layers.", "layers_"): v for k, v in state_dict.items()}
        missing, unexpected = encoder.load_state_dict(converted_sd, strict=False)

        print(f"SwinV2-L checkpoint loaded: {ckpt_path}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    if freeze_backbone:
        for p in encoder.parameters():
            p.requires_grad = False
        print("SwinV2-L backbone is frozen.")

    return encoder


def build_model(num_classes, input_size, output_size, drop_rate, pretrained_path=None, freeze_backbone=False):
    encoder = build_encoder(drop_rate, pretrained_path, freeze_backbone)

    return SCDNet(
        encoder=encoder,
        in_channel_list=[192, 384, 768, 1536],
        num_classes=num_classes,
        output_size=output_size,
        drop_rate=drop_rate,
        out_channels=256,
        channel_last=True,
        use_refinement_block=True,
    )