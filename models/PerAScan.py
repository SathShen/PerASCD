# finetune wrapper
import torch.nn as nn
import torch
from .pera_layers.vit_adapter import DinoV2_ViTAdapter
import copy
from .pera_layers.hrscd_decoder_for_pera import Decoder4PerA as Decoder
import torch.nn.functional as F
import math

from utils.misc import initialize_weights
from models.CSWin_Transformer import mit

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low//scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low//scale_ratio),
            nn.BatchNorm2d(in_channels_low//scale_ratio),
            nn.ReLU(inplace=True) )
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x
    
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ECAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ECAConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        t = int(abs((math.log(in_channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1

        self.conv1x1 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            padding=(k - 1) // 2,
            bias=False
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x)  # b, c, 1, 1
        y = y.view(b, 1, c)
        y = self.conv1x1(y)  # b, 1, c
        y = y.view(b, c, 1, 1)
        y = torch.sigmoid(y)

        x = x * y.expand_as(x)
        x = self.conv(x)
        return x


class FPN(nn.Module):
    """特征金字塔网络，将多尺度特征融合到统一分辨率"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        # 处理每个层级的1x1卷积，统一通道数
        self.lateral_convs = nn.ModuleList([
            # ECAConv2d(in_channels, out_channels, 1)
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        
        # 融合卷积
        self.fusion_conv_low = nn.Sequential(
            conv1x1(out_channels * 2, out_channels // 2),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            conv3x3(out_channels // 2, out_channels // 2),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.fusion_conv = nn.Sequential(
            conv1x1(out_channels * 2, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, inputs):
        # inputs: [feat0, feat1, feat2, feat3] 分辨率递减 1/4 -> 1/8 -> 1/16 -> 1/32
        laterals = [
            lateral_conv(inputs[i]) 
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        feat0 = laterals[0]         # 1/4
        feat1 = F.interpolate(laterals[1], scale_factor=2, mode='nearest') # 1/8 -> 1/4
        feat2 = F.interpolate(laterals[2], scale_factor=2, mode='nearest') # 1/16 -> 1/8
        feat3 = F.interpolate(laterals[3], scale_factor=4, mode='nearest') # 1/32 -> 1/8

        feat_fuse0 = self.fusion_conv_low(torch.cat([feat0, feat1], dim=1))     # 1/4
        feat_fuse1 = self.fusion_conv(torch.cat([feat2, feat3], dim=1))     # 1/8
    
        return feat_fuse0, feat_fuse1
    

class PerAScan(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=7, 
                 input_size=448,
                 output_size=512,
                 arch='ViT-B/16',
                 droppath=0., 
                 pretrained_pera_path=None, 
                 is_distilled_pera=False,
                 is_freeze_backbone=False
                 ):
        super(PerAScan, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.droppath = droppath

        self.in_channels = in_channels
        if arch == 'ViT-B/16':
            embed_dim = 768
        elif arch == 'ViT-G/16/1024':
            embed_dim = 1024
        else:
            raise NotImplementedError(f"{arch} is not implemented yet.")

        # fuse module
        self.fpn_t1 = FPN([embed_dim] * 4, 128)  # 输入4个特征层的通道数，输出统一通道数
        
        # 时相2的FPN  
        self.fpn_t2 = FPN([embed_dim] * 4, 128)
        # fuse module
        

        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.transformer = mit(img_size=output_size//4, in_chans=128*3, embed_dim=128*3)
        
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1  = _DecoderBlock(128, 64,  128)
        self.Dec2  = _DecoderBlock(128, 64,  128)
        
        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
            
        initialize_weights(self.Dec1, self.Dec2, self.DecCD, self.classifierA, self.classifierB, self.classifierCD)

        
        self.backbone = self.build_backbone(arch=arch, 
                                            pretrained_pera_path=pretrained_pera_path, 
                                            is_distilled_pera=is_distilled_pera, 
                                            is_freeze_backbone=is_freeze_backbone
                                            )
        # self.backbone_cd = self.build_backbone(arch=arch, 
        #                                         pretrained_pera_path=pretrained_pera_path, 
        #                                         is_distilled_pera=is_distilled_pera, 
        #                                         is_freeze_backbone=is_freeze_backbone,
        #                                         is_cd=True)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def build_backbone(self, arch='ViT-B/16', pretrained_pera_path=None, is_distilled_pera=False, is_freeze_backbone=False, is_cd=False):
        if arch == 'ViT-B/16':
            backbone = DinoV2_ViTAdapter(img_size=self.input_size,
                            patch_size=16, 
                            embed_dim=768,
                            depth=12,
                            num_heads=12,
                            mlp_ratio=4.0,
                            drop_path_rate=self.droppath,
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
                                            drop_path_rate=self.droppath,
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
            print('Backbone is frozen!')
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
            del model_dict
            print(f"Pretrained model loaded from {pretrained_pera_path}")
        return backbone

    def forward(self, x1, x2):
        outputs_1 = self.backbone(x1)
        outputs_2 = self.backbone(x2)

        # fpn_outputs_t1[0]是1/4分辨率，fpn_outputs_t1[1]是1/8分辨率，以此类推
        fpn_outputs_t1 = self.fpn_t1(outputs_1)
        fpn_outputs_t2 = self.fpn_t2(outputs_2)
        
        # 使用最高分辨率的特征（1/4）作为主要特征
        x1_low = fpn_outputs_t1[0]  # 1/4分辨率，64通道  
        x2_low = fpn_outputs_t2[0]  # 1/4分辨率，64通道  # b, 64, 128, 128

        
        # 使用次高分辨率的特征（1/8）作为低层特征提供给解码器
        x1_fused = fpn_outputs_t1[1]  # 1/8分辨率，128通道  # b, 128, 64, 64
        x2_fused = fpn_outputs_t2[1]  # 1/8分辨率，128通道  

        xc = self.resCD(torch.cat([x1_fused, x2_fused], 1))

        x1 = self.Dec1(x1_fused, x1_low)
        x2 = self.Dec2(x2_fused, x2_low)        
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)
                
        x = torch.cat([x1, x2, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]
        
        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        change = self.classifierCD(xc)

        outsize = (self.output_size, self.output_size)
        return F.interpolate(change, outsize, mode='bilinear', align_corners=False), \
               F.interpolate(out1, outsize, mode='bilinear', align_corners=False), \
               F.interpolate(out2, outsize, mode='bilinear', align_corners=False)