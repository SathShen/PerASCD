# finetune wrapper
import torch.nn as nn
import torch
from .pera_layers.vit_adapter import DinoV2_ViTAdapter
import copy
from .pera_layers.hrscd_decoder_for_pera import Decoder4PerA as Decoder

class PerASCD(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=7, 
                 input_size=448,
                 output_size=512,
                 arch='ViT-B/16',
                 head_name='HRSCD4', 
                 pretrained_pera_path=None, 
                 is_distilled_pera=False,
                 is_freeze_backbone=False
                 ):
        super(PerASCD, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.in_channels = in_channels
        if arch == 'ViT-B/16':
            embed_dim = 768
        elif arch == 'ViT-G/16/1024':
            embed_dim = 1024
        else:
            raise NotImplementedError(f"{arch} is not implemented yet.")

        self.head_name = head_name
        if self.head_name == 'HRSCD4':
            self.TCDecoder = Decoder(num_classes, depths=embed_dim, output_size=output_size)
            self.CDDecoder = Decoder(1, depths=embed_dim, CD=True, output_size=output_size)
        else:
            raise NotImplementedError(f"{head_name} is not implemented yet.")
        
        self.backbone = self.build_backbone(arch=arch, 
                                            pretrained_pera_path=pretrained_pera_path, 
                                            is_distilled_pera=is_distilled_pera, 
                                            is_freeze_backbone=is_freeze_backbone
                                            )
        self.backbone_cd = self.build_backbone(arch=arch, 
                                                pretrained_pera_path=pretrained_pera_path, 
                                                is_distilled_pera=is_distilled_pera, 
                                                is_freeze_backbone=is_freeze_backbone,
                                                is_cd=True)

    def build_backbone(self, arch='ViT-B/16', pretrained_pera_path=None, is_distilled_pera=False, is_freeze_backbone=False, is_cd=False):
        if arch == 'ViT-B/16':
            backbone = DinoV2_ViTAdapter(img_size=self.input_size,
                            patch_size=16, 
                            embed_dim=768,
                            depth=12,
                            num_heads=12,
                            mlp_ratio=4.0,
                            drop_path_rate=0.3,
                            drop_path_uniform=True,
                            conv_inplane=64,
                            n_points=4,
                            deform_num_heads=12,
                            init_values=0.0,
                            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                            with_cffn=True,
                            cffn_ratio=0.25,
                            add_vit_feature=True,
                            use_extra_extractor=True,
                            with_cp=False,
                            in_chans=6 if is_cd else 3)
        elif arch == 'ViT-G/16/1024':
            # setting of ViT-G/16/1024
            backbone = DinoV2_ViTAdapter(img_size=self.input_size,
                                            patch_size=16, 
                                            embed_dim=1024,
                                            depth=40,
                                            num_heads=16,
                                            mlp_ratio=4.0,
                                            drop_path_rate=0.3,
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
                                            in_chans=6 if is_cd else 3)
                
        backbone.head = None
        backbone.mask_token = None

        if pretrained_pera_path is not None:
            backbone = self.load_dict_to_backbone(backbone, pretrained_pera_path, distilled=is_distilled_pera, is_cd=is_cd)

        if is_freeze_backbone:
            for param in backbone.blocks.parameters():
                param.requires_grad = False
            for param in backbone.patch_embed.parameters():
                param.requires_grad = False
            backbone.pos_embed.requires_grad = False
            backbone.cls_token.requires_grad = False
            backbone.norm.requires_grad = False
        return backbone


    def load_dict_to_backbone(self, backbone,pretrained_pera_path, distilled=False, is_cd=False):
        if distilled:
            model_dict = torch.load(pretrained_pera_path, map_location='cpu', weights_only=False)
            model_dict = {k.replace('student.backbone.', ''): v for k, v in model_dict['model'].items() if k.startswith('student.backbone.')}
            if is_cd:
                # remove patch_embed
                model_dict.pop('patch_embed.proj.weight', None)
                model_dict.pop('patch_embed.proj.bias', None)
                model_dict.pop('pos_embed', None)
            backbone.load_state_dict(model_dict, strict=False)
            del model_dict
            print(f"Pretrained model loaded from {pretrained_pera_path}")
        else:
            model_dict = torch.load(pretrained_pera_path, map_location='cpu', weights_only=False)
            model_dict = {k.replace('teacher.backbone.', ''): v for k, v in model_dict['model'].items() if k.startswith('teacher.backbone.')}
            if is_cd:
                # remove patch_embed
                model_dict.pop('patch_embed.proj.weight', None)
                model_dict.pop('patch_embed.proj.bias', None)
                model_dict.pop('pos_embed', None)
            backbone.load_state_dict(model_dict, strict=False)
        return backbone

    def forward(self, imgsA, imgsB):
        outputs_1 = self.backbone(imgsA)
        outputs_2 = self.backbone(imgsB)
        if self.head_name == 'HRSCD4':
            tc1 = self.TCDecoder(outputs_1)
            tc2 = self.TCDecoder(outputs_2)

            # Change Detection
            outputs_cd = self.backbone_cd(torch.cat((imgsA, imgsB), 1))
            for i in range(len(outputs_cd)):
                outputs_cd[i] = torch.cat((outputs_cd[i], torch.abs(outputs_1[i] - outputs_2[i])), 1)
            cm = self.CDDecoder(outputs_cd)

            return cm, tc1, tc2