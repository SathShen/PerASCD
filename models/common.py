import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAMconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction=16):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        avg_out = self.mlp1(self.avg_pool(x))
        max_out = self.mlp1(self.max_pool(x))
        channel_w = self.sigmoid(avg_out + max_out)
        x = x * channel_w

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_w = self.sigmoid(self.conv1x1(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_w

        return self.conv2d(x)


class ChangeAwareGatingModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv_local = nn.Conv2d(in_channels // 4, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels // 4, 2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        avg = self.avg_pool(x)
        avg = self.conv_global(avg)
        global_weight = self.sigmoid(avg)

        logit = self.conv_local(x)
        local_weight = self.sigmoid(logit)

        return local_weight * (1 + global_weight)


class CascadeGatedBlock(nn.Module):
    def __init__(self, feat_channels, out_channels, drop_rate=0.0, use_lateral=True):
        super().__init__()

        self.use_lateral = use_lateral

        self.feat_conv0 = nn.Sequential(
            CBAMconv2d(feat_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.feat_convc = nn.Sequential(
            CBAMconv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.highconv0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.highconv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.highconvc = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.lowconv0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.lowconv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.lowconvc = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.cagm = ChangeAwareGatingModule(out_channels * 2)
        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x0, x1, xc, feat0=None, feat1=None):
        x0 = self.highconv0(x0)
        x1 = self.highconv1(x1)
        xc = self.highconvc(torch.cat([xc, x0, x1], dim=1))

        if self.use_lateral:
            if feat0 is None or feat1 is None:
                raise ValueError("use_lateral=True requires feat0 and feat1.")

            x0 = F.interpolate(x0, scale_factor=2, mode="bilinear", align_corners=False)
            x1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=False)
            xc = F.interpolate(xc, scale_factor=2, mode="bilinear", align_corners=False)

            f0 = self.feat_conv0(feat0)
            f1 = self.feat_conv0(feat1)
        else:
            f0 = self.feat_conv0(x0)
            f1 = self.feat_conv0(x1)

        fc = self.feat_convc(torch.abs(f0 - f1))

        hardship_map = self.cagm(torch.cat([xc, fc], dim=1))

        w_high = hardship_map[:, 0].unsqueeze(1)
        w_low = hardship_map[:, 1].unsqueeze(1)

        x0 = self.lowconv0((w_high * x0) + (w_low * f0))
        x1 = self.lowconv1((w_high * x1) + (w_low * f1))
        xc = self.lowconvc((w_high * xc) + (w_low * fc))

        x0 = self.dropout(x0)
        x1 = self.dropout(x1)
        xc = self.dropout(xc)

        return x0, x1, xc


class CascadeGatedDecoder(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channels,
        drop_rate=0.0,
        use_refinement_block=False,
    ):
        super().__init__()

        self.use_refinement_block = use_refinement_block

        self.first_feat_conv0 = nn.Sequential(
            CBAMconv2d(in_channel_list[-1], out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        fusion_blocks = []
        for i in range(len(in_channel_list) - 1):
            fusion_blocks.append(
                CascadeGatedBlock(
                    feat_channels=in_channel_list[len(in_channel_list) - i - 2],
                    out_channels=out_channels,
                    drop_rate=drop_rate,
                    use_lateral=True,
                )
            )

        self.fusion_blocks = nn.ModuleList(fusion_blocks)

        if use_refinement_block:
            self.refinement_block = CascadeGatedBlock(
                feat_channels=out_channels,
                out_channels=out_channels,
                drop_rate=drop_rate,
                use_lateral=False,
            )
        else:
            self.refinement_block = None

    def forward(self, feat_list_a, feat_list_b):
        x0 = self.first_feat_conv0(feat_list_a[-1])
        x1 = self.first_feat_conv0(feat_list_b[-1])
        xc = torch.abs(x0 - x1)

        for i, block in enumerate(self.fusion_blocks):
            feat0 = feat_list_a[len(feat_list_a) - i - 2]
            feat1 = feat_list_b[len(feat_list_b) - i - 2]
            x0, x1, xc = block(x0, x1, xc, feat0, feat1)

        if self.refinement_block is not None:
            x0, x1, xc = self.refinement_block(x0, x1, xc)

        return x0, x1, xc


class SCDNet(nn.Module):
    def __init__(
        self,
        encoder,
        in_channel_list,
        num_classes,
        output_size,
        drop_rate=0.0,
        out_channels=256,
        channel_last=False,
        use_refinement_block=False,
    ):
        super().__init__()

        self.encoder = encoder
        self.output_size = output_size
        self.channel_last = channel_last

        self.decoder = CascadeGatedDecoder(in_channel_list, out_channels, drop_rate, use_refinement_block)
        self.classifier_a = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        self.classifier_b = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        self.classifier_cd = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, 1, kernel_size=1),
        )

        self.initialize_weights(self.decoder, self.classifier_a, self.classifier_b, self.classifier_cd)

    def _encode(self, x):
        feats = self.encoder(x)
        if self.channel_last:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        return feats

    def forward(self, img_a, img_b):
        feat_a = self._encode(img_a)
        feat_b = self._encode(img_b)

        x0, x1, xc = self.decoder(feat_a, feat_b)

        out_a = self.classifier_a(x0)
        out_b = self.classifier_b(x1)
        out_cd = self.classifier_cd(xc)

        outsize = (self.output_size, self.output_size)

        return (
            F.interpolate(out_cd, outsize, mode="bilinear", align_corners=False),
            F.interpolate(out_a, outsize, mode="bilinear", align_corners=False),
            F.interpolate(out_b, outsize, mode="bilinear", align_corners=False),
        )
    
    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()