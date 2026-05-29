import torch
import torch.nn as nn

from models.pera_layers.vit_adapter import DinoV2_ViTAdapter
from .common import SCDNet


DEFAULT_CKPT = "/data2/sht/Outputs/Backup/Pretrain/ViTGall22601_250221020940/Run2/autosave/pera_ViTGall22601_ep42_auto.params"


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

        ckpt_path = pretrained_path or DEFAULT_CKPT
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

        missing, unexpected = self.backbone.load_state_dict(teacher_state, strict=False)
        print(f"PerA DINOv2-G checkpoint loaded: {pretrained_path}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("PerA DINOv2-G backbone is frozen.")

    def forward(self, x):
        return self.backbone(x)


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

    return SCDNet(
        encoder=encoder,
        in_channel_list=[1024, 1024, 1024, 1024],
        num_classes=num_classes,
        output_size=output_size,
        drop_rate=drop_rate,
        out_channels=256,
        channel_last=False,
        use_refinement_block=True,
    )