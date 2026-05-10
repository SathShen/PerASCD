import torch
import torch.nn as nn
import os
import argparse
import timm
import torch.nn.functional as F
from torchvision.models import resnet50
from models.SatMAE_temporal import get_1d_sincos_pos_embed_from_grid_torch, mae_vit_large_patch16
from timm.models.layers import DropPath

class CBAMconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction=16):
        super(CBAMconv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        # spatial attention
        self.conv1x1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        # channel attention
        avg_out = self.mlp1(self.avg_pool(x))
        max_out = self.mlp1(self.max_pool(x))
        channel_w = self.sigmoid(avg_out + max_out)
        x = x * channel_w

        # spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_w = self.sigmoid(self.conv1x1(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_w

        x = self.conv2d(x)
        return x
    


# PyTorch-like code snippet for the Gating Module
class ChangeAwareGatingModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // 4, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels // 4, 2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        logit = self.conv2(x)
        avg = self.avg_pool(x)
        avg = self.conv_global(avg)
        change_avg = self.sigmoid(avg)
        change_map = self.sigmoid(logit) * (1 + change_avg)
        return change_map



class CascadeGatedBlock(nn.Module):
    def __init__(self, feat_channels, out_channels, drop_rate=0.):
        super().__init__()
        self.feat_conv0 = nn.Sequential(CBAMconv2d(feat_channels, out_channels, kernel_size=3),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        self.feat_convc = nn.Sequential(CBAMconv2d(out_channels, out_channels, kernel_size=3),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
                                        
        self.highconv0 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        self.highconv1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        self.highconvc = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        
        self.lowconv0 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        self.lowconv1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        self.lowconvc = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        
        self.cagm = ChangeAwareGatingModule(out_channels * 2)

        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x0, x1, xc, feat0=None, feat1=None):
        # feat0, feat1: [B, oc, scale, scale]
        # x0, x1, xc: [B, oc, scale // 2, scale // 2]

        # 语义提取
        x0 = self.highconv0(x0) # [B, oc, scale // 2, scale // 2] 
        x1 = self.highconv1(x1) 
        xc = self.highconvc(torch.cat([xc, x0, x1], dim=1)) # [B, oc, scale // 2, scale // 2]

        if feat0 is not None and feat1 is not None:
            x0 = nn.functional.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=False) # [B, oc, scale, scale]
            x1 = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False) # [B, oc, scale, scale]
            xc = nn.functional.interpolate(xc, scale_factor=2, mode='bilinear', align_corners=False) # [B, oc, scale, scale]
        
        # 特征融合
        if feat0 is not None and feat1 is not None:
            f0 = feat0
            f1 = feat1
        else:
            f0 = x0
            f1 = x1

        f0 = self.feat_conv0(f0)  # [B, oc, scale, scale]
        f1 = self.feat_conv0(f1)
        fc = self.feat_convc(torch.abs(f0 - f1))
        # mask = torch.sigmoid(fc.mean(dim=1, keepdim=True))

        hardship_map = self.cagm(torch.cat([xc, fc], dim=1))  # [B, 1, scale // 2, scale // 2]

        w_high = hardship_map[:, 0].unsqueeze(1)
        w_low = hardship_map[:, 1].unsqueeze(1)

        x0 = self.lowconv0((w_high * x0) + (w_low * f0)) # [B, oc, scale, scale]
        x1 = self.lowconv1((w_high * x1) + (w_low * f1))
        xc = self.lowconvc((w_high * xc) + (w_low * fc)) # [B, oc, scale, scale]

        x0 = self.dropout(x0)
        x1 = self.dropout(x1)
        xc = self.dropout(xc)

        return x0, x1, xc

    
class CascadeGatedDecoder(nn.Module):
    def __init__(self, in_channel_list, out_channels, num_blocks, drop_rate=0.):
        super().__init__()
        self.num_blocks = num_blocks
        self.first_feat_conv0 = nn.Sequential(CBAMconv2d(in_channel_list[-1], out_channels, kernel_size=3),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([
            CascadeGatedBlock(in_channel_list[len(in_channel_list) - i - 2] if i < len(in_channel_list) - 1 else out_channels, 
                         out_channels, 
                         drop_rate
                        #  feat_input=True if i < len(in_channel_list) - 1 else False
                         )
            for i in range(num_blocks)
        ])
    
    def forward(self, feat_listA, feat_listB):
        # inputs: [feat0, feat1, feat2, feat3] 分辨率递减 1/4 -> 1/8 -> 1/16 -> 1/32
        x0 = self.first_feat_conv0(feat_listA[-1])
        x1 = self.first_feat_conv0(feat_listB[-1])
        xc = torch.abs(x0 - x1)
        for i, block in enumerate(self.blocks):
            if i < len(feat_listA) - 1:
                feat0 = feat_listA[len(feat_listA)-i-2]
                feat1 = feat_listB[len(feat_listA)-i-2]
            else:
                feat0 = None
                feat1 = None
            x0, x1, xc = block(x0, x1, xc, feat0, feat1)
        return x0, x1, xc


def add_drop_path_to_resnet(model, drop_path_rate=0.3):
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    
    total_blocks = sum(len(layer) for layer in layers)
    block_idx = 0

    for layer in layers:
        for block in layer:
            drop_rate = drop_path_rate * block_idx / (total_blocks - 1)
            block_idx += 1

            block.drop_path = DropPath(drop_rate) if drop_rate > 0 else nn.Identity()

            # 保存原 forward
            old_forward = block.forward

            # 定义新 forward
            def new_forward(self, x, old_forward=old_forward):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if hasattr(self, 'conv3'):  # Bottleneck
                    out = self.relu(out)
                    out = self.conv3(out)
                    out = self.bn3(out)

                # 👉 DropPath 加在 residual branch
                out = self.drop_path(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out

            block.forward = new_forward.__get__(block, block.__class__)

    return model

def build_net(encoder_name, num_classes, output_size, drop_rate):
    if encoder_name == 'resnet50':
        base = resnet50(pretrained=True)
        base = add_drop_path_to_resnet(base, drop_path_rate=drop_rate)
        in_channel_list = [256, 512, 1024, 2048]
        class Encoder(nn.Module):
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
        encoder = Encoder(base)
    elif encoder_name == 'vmambaB':
        from models.vmamba import Backbone_VSSM  # 或 from vmamba import Backbone_VSSM
        base = Backbone_VSSM(
            patch_size=4,
            in_chans=3,
            depths=[2, 2, 15, 2],           # Base 配置（或 [2,2,20,2] 更深）
            dims=[128, 256, 512, 1024],     # in_channel_list
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",         # 高效版（無 z 殘差，速度快）
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=drop_rate,
            patch_norm=True,
            norm_layer="ln2d",              # channel first (NCHW)
            downsample_version="v3",        # overlap conv downsample，效果好
            patchembed_version="v2",
            use_checkpoint=False,           # 記憶體夠可改 True 省 mem
            out_indices=(0, 1, 2, 3),       # 輸出 4 階段特徵
        )

        # 通道數自動
        ckpt_path = '/data2/sht/checkpoints/vmamba/vssm_base_0229_ckpt_epoch_237.pth'
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        base.load_state_dict(state_dict, strict=False)
        print(f"VMamba Base 权重加载成功: {ckpt_path}")
        in_channel_list = base.dims  # [128, 256, 512, 1024]
        encoder = base
    elif encoder_name == 'swinV2B':
        base = timm.create_model(
            # 'swinv2_large_window12to16_192to256',
            'swinv2_base_window8_256',
            pretrained=False,
            features_only=True,
            img_size=512,
            drop_path_rate=drop_rate
        )

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        converted_sd = {}
        for k, v in state_dict.items():
            new_k = k.replace('layers.', 'layers_')   # 只這一行就夠了
            converted_sd[new_k] = v

        # 再載入（因為 features_only=True，可能還有些 head 的 key 多出來）
        missing, unexpected = base.load_state_dict(converted_sd, strict=False)
        print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        # base.load_state_dict(state_dict, strict=True)
        in_channel_list = base.feature_info.channels()  # [192, 384, 768, 1536]
        encoder = base
    elif encoder_name == 'swinV2L':
        base = timm.create_model(
            'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
            pretrained=True,
            features_only=True,
            img_size=512,
            drop_path_rate=drop_rate,
        )
        from safetensors.torch import load_file
        checkpoint_path = '/data2/sht/checkpoints/swinv2/swinv2/swinv2_large_window12to16_192to256.ms_in22k_ft_in1k.safetensors'
        state_dict = load_file(checkpoint_path)
        converted_sd = {k.replace('layers.', 'layers_'): v for k, v in state_dict.items()}
        missing, unexpected = base.load_state_dict(converted_sd, strict=False)

        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

        in_channel_list = base.feature_info.channels()  # [192, 384, 768, 1536]
        encoder = base
        print("SwinV2-Large safetensors 權重載入成功，從本地檔:", checkpoint_path)

    elif encoder_name == 'SatMAE':
        # 1. 实例化模型 (img_size=512)
        base = mae_vit_large_patch16(img_size=512, drop_path=drop_rate)
        
        # 2. 加载预训练权重
        ckpt_path = '/data2/sht/checkpoints/SatMAE/SatMAE_pretrain_fmow_temporal.pth' # 请确保路径正确
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

            # ---------------------------------------------------------------------
            # [关键修复 1]: 调整 Encoder 的 pos_embed 形状以匹配权重
            # 这样 load_state_dict 不会报错，我们稍后在 forward 里做插值
            # ---------------------------------------------------------------------
            if 'pos_embed' in state_dict and state_dict['pos_embed'].shape != base.pos_embed.shape:
                print(f"调整 Encoder pos_embed: {base.pos_embed.shape} -> {state_dict['pos_embed'].shape}")
                base.pos_embed = nn.Parameter(torch.zeros(state_dict['pos_embed'].shape))

            # ---------------------------------------------------------------------
            # [关键修复 2]: 删除 Decoder 相关的权重
            # 下游任务不需要 decoder，且 decoder_pos_embed 尺寸不匹配会导致 Crash
            # ---------------------------------------------------------------------
            keys_to_remove = [k for k in state_dict.keys() if 'decoder' in k or 'mask_token' in k]
            for k in keys_to_remove:
                # print(f"Removing key: {k}") # debug用
                del state_dict[k]
            
            # 3. 加载权重
            msg = base.load_state_dict(state_dict, strict=False)
            print(f"SatMAE weights loaded: {msg}")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, using random init.")

        # SatMAE ViT-Large embedding dim
        in_channel_list = [1024, 1024, 1024, 1024]

        class SatMAE_Encoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.out_indices = [5, 11, 17, 23]
                
                # 预先计算固定的时间嵌入 (T=0)
                dummy_time = torch.zeros(1, 3)
                t_emb = torch.cat([
                    get_1d_sincos_pos_embed_from_grid_torch(128, dummy_time[:, 0]),
                    get_1d_sincos_pos_embed_from_grid_torch(128, dummy_time[:, 1]),
                    get_1d_sincos_pos_embed_from_grid_torch(128, dummy_time[:, 2])
                ], dim=1).float() 
                self.register_buffer('fixed_time_embed', t_emb.unsqueeze(0)) # [1, 1, 384]

            def forward(self, x):
                # x: [B, 3, 512, 512]
                B, C, H, W = x.shape
                
                # Patch Embed -> [B, 1024, 1024]
                x = self.model.patch_embed(x)
                
                # --- 动态 Pos Embed 插值逻辑 ---
                # 取出权重中的 pos_embed (基于224的, 196 patches)
                spatial_pos = self.model.pos_embed[:, 1:, :] 
                num_patches_current = x.shape[1] # 1024 patches (基于512的)
                
                if num_patches_current != spatial_pos.shape[1]:
                    gs_old = int(spatial_pos.shape[1] ** 0.5) # 14
                    gs_new = int(num_patches_current ** 0.5)  # 32
                    
                    # [1, 196, 640] -> [1, 640, 14, 14]
                    sp = spatial_pos.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                    # Interpolate 14 -> 32
                    sp = F.interpolate(sp, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
                    # -> [1, 1024, 640]
                    spatial_pos = sp.permute(0, 2, 3, 1).reshape(1, num_patches_current, -1)
                
                # 拼接时间嵌入
                time_pos = self.fixed_time_embed.expand(B, num_patches_current, -1)
                spatial_pos = spatial_pos.expand(B, -1, -1)
                full_pos = torch.cat([spatial_pos, time_pos], dim=-1) # [B, 1024, 1024]
                
                x = x + full_pos

                # --- Backbone Forward ---
                features = []
                for i, blk in enumerate(self.model.blocks):
                    x = blk(x)
                    if i in self.out_indices:
                        h_grid = H // 16
                        w_grid = W // 16
                        # [B, L, D] -> [B, D, H/16, W/16]
                        out = x.permute(0, 2, 1).reshape(B, -1, h_grid, w_grid)
                        features.append(out)
                
                # --- Feature Pyramid ---
                f0 = F.interpolate(features[0], scale_factor=4, mode='bilinear', align_corners=False)
                f1 = F.interpolate(features[1], scale_factor=2, mode='bilinear', align_corners=False)
                f2 = features[2]
                f3 = F.avg_pool2d(features[3], kernel_size=2, stride=2)
                
                return [f0, f1, f2, f3]

        encoder = SatMAE_Encoder(base)

    elif encoder_name == 'SeCo':
        base = resnet50(pretrained=False)
        base = add_drop_path_to_resnet(base, drop_path_rate=drop_rate)
        base = torch.nn.Sequential(*list(base.children())[:-2])

        ckpt_path = '/data2/sht/checkpoints/SeCo/seco_resnet50_1m_converted.pth'

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SeCo 權重不存在: {ckpt_path}")
        print(f"載入 SeCo 權重: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = base.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys: {missing}")
        print(f"  Unexpected keys: {unexpected}")

        class SeCoEncoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                # Stem: Conv1 -> BN -> ReLU -> MaxPool
                x = self.model[0](x)
                x = self.model[1](x)
                x = self.model[2](x)
                x = self.model[3](x)

                # Layers
                f1 = self.model[4](x)  # Layer 1: [B, 256, H/4, W/4]
                f2 = self.model[5](f1) # Layer 2: [B, 512, H/8, W/8]
                f3 = self.model[6](f2) # Layer 3: [B, 1024, H/16, W/16]
                f4 = self.model[7](f3) # Layer 4: [B, 2048, H/32, W/32]

                return [f1, f2, f3, f4]

        encoder = SeCoEncoder(base)
        in_channel_list = [256, 512, 1024, 2048] # 对应 layer1 ~ layer4
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")

    out_channels = 256  # Common choice for unified channels in decoder
    num_blocks = len(in_channel_list)
    decoder = CascadeGatedDecoder(in_channel_list, out_channels, num_blocks, drop_rate)

    class Net(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.classifierA = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            self.classifierB = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            self.classifierCD = nn.Sequential(nn.Conv2d(out_channels, out_channels // 2, kernel_size=1), 
                                              nn.BatchNorm2d(out_channels // 2), 
                                              nn.ReLU(), 
                                              nn.Conv2d(out_channels // 2, 1, kernel_size=1))

        def forward(self, imgA, imgB):
            feat_listA = self.encoder(imgA)
            feat_listB = self.encoder(imgB)
            if encoder_name.lower().startswith('swinv2'):
                feat_listA = [f.permute(0, 3, 1, 2).contiguous() for f in feat_listA]
                feat_listB = [f.permute(0, 3, 1, 2).contiguous() for f in feat_listB]
            x0, x1, xc = self.decoder(feat_listA, feat_listB)
            x0 = self.classifierA(x0)
            x1 = self.classifierB(x1)
            xc = self.classifierCD(xc)
            outsize = (output_size, output_size)
            return F.interpolate(xc, outsize, mode='bilinear', align_corners=False), \
                F.interpolate(x0, outsize, mode='bilinear', align_corners=False), \
                F.interpolate(x1, outsize, mode='bilinear', align_corners=False)

    return Net(encoder, decoder)