import torch
from collections import OrderedDict
import torch.nn as nn

from models.pera_layers.vit_adapter import DinoV2_ViTAdapter
from .common import SCDNet


class PerAEncoder(nn.Module):
    def __init__(
        self,
        input_size=448,
        drop_rate=0.0,
        pretrained_path=None,
        freeze_backbone=False,
    ):
        super().__init__()

        self.backbone = DinoV2_ViTAdapter(
            img_size=input_size,
            patch_size=16,
            embed_dim=1024,
            depth=40,
            num_heads=16,
            mlp_ratio=4.0,
            drop_path_rate=drop_rate,
            drop_path_uniform=True,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            init_values=0.0,
            interaction_indexes=[[0, 9], [10, 19], [20, 29], [30, 39]],
            with_cffn=True,
            cffn_ratio=0.25,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=False,
            in_chans=3,
        )

        self.backbone.head = None
        self.backbone.mask_token = None

        ckpt_path = pretrained_path
        if ckpt_path is not None:
            self.load_pretrained(ckpt_path)

        if freeze_backbone:
            self.freeze_backbone()

    def load_pretrained(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        teacher_state = {
            k.replace("teacher.backbone.", ""): v
            for k, v in state_dict.items()
            if k.startswith("teacher.backbone.")
        }

        if not teacher_state:
            raise ValueError("Expected teacher.backbone.* foundation weights; full weights use --finetune-path")
        teacher_state = {k: v for k, v in teacher_state.items()
                         if not k.startswith("head.") and k != "mask_token"}
        missing, unexpected = self.backbone.load_state_dict(teacher_state, strict=False)
        adapter_keys = ("spm.", "interactions.", "up.", "norm1.", "norm2.", "norm3.", "norm4.")
        bad = [k for k in missing if k != "level_embed" and not k.startswith(adapter_keys)]
        if bad or unexpected:
            raise ValueError(f"Foundation checkpoint mismatch: missing={bad}, unexpected={unexpected}")
        print(f"PerA DINOv2-G checkpoint loaded: {pretrained_path}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("PerA DINOv2-G backbone is frozen.")

    def forward(self, x):
        return self.backbone(x)


class PerASCD(SCDNet):
    """Fixed three-block model; accept parameter names from the existing Docker run."""
    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError("PerASCD requires strict checkpoint loading")
        prefixes = (("backbone.", "encoder.backbone."),
                    ("decoder.blocks.", "decoder.fusion_blocks."),
                    ("classifierA.", "classifier_a."),
                    ("classifierB.", "classifier_b."),
                    ("classifierCD.", "classifier_cd."))
        mapped = OrderedDict()
        for key, value in state_dict.items():
            key = key.removeprefix("module.")
            for old, new in prefixes:
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            if key in mapped:
                raise ValueError("Checkpoint key collision: " + key)
            mapped[key] = value
        return super().load_state_dict(mapped, strict=True, assign=assign)


def build_model(
    num_classes,
    input_size,
    output_size,
    drop_rate,
    pretrained_path=None,
    freeze_backbone=False,
):
    encoder = PerAEncoder(
        input_size=input_size,
        drop_rate=drop_rate,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone,
    )

    return PerASCD(
        encoder=encoder,
        in_channel_list=[1024, 1024, 1024, 1024],
        num_classes=num_classes,
        output_size=output_size,
        drop_rate=drop_rate,
        out_channels=128,
        channel_last=False,
        use_refinement_block=False,
    )