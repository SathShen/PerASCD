import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_ss(nn.Module):

    def __init__(self, inplanes, planes=None, subsamp=1):
        super(BasicBlock_ss, self).__init__()
        if planes == None:
            planes = inplanes * subsamp
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.subsamp = subsamp
        self.doit = planes != inplanes
        if self.doit:
            self.couple = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        if self.doit:
            residual = self.couple(x)
            residual = self.bnc(residual)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.subsamp > 1:
            out = F.max_pool2d(out, kernel_size=self.subsamp, stride=self.subsamp)
            residual = F.max_pool2d(residual, kernel_size=self.subsamp, stride=self.subsamp)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_us(nn.Module):

    def __init__(self, inplanes, upsamp=1):
        super(BasicBlock_us, self).__init__()
        planes = int(inplanes / upsamp)  # assumes integer result, fix later
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1,
                                        bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsamp = upsamp
        self.couple = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1,
                                         bias=False)
        self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.couple(x)
        residual = self.bnc(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Decoder4PerA(nn.Module):
    """Decoder block"""

    def __init__(self, label_nbr, depths, CD=False, output_size=512):
        """Init Decoder fields."""
        super(Decoder4PerA, self).__init__()

        self.output_size = output_size

        channels = [3, 8, 16, 32, 64, 128, 256]

        cur_depth = channels[6]

        # Decoding stage 5
        self.decres5_1 = BasicBlock_ss(depths + CD * depths, planes=cur_depth)
        self.decres5_2 = BasicBlock_ss(cur_depth)
        self.decres5_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = channels[5]

        # Decoding stage 4
        self.decres4_1 = BasicBlock_ss(cur_depth + (1 + CD) * depths, planes=cur_depth)
        self.decres4_2 = BasicBlock_ss(cur_depth)
        self.decres4_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = channels[4]

        # Decoding stage 3
        self.decres3_1 = BasicBlock_ss(cur_depth + (1 + CD) * depths, planes=cur_depth)
        self.decres3_2 = BasicBlock_ss(cur_depth)
        self.decres3_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = channels[3]

        # Decoding stage 2
        self.decres2_1 = BasicBlock_ss(cur_depth + (1 + CD) * depths, planes=cur_depth)
        self.decres2_2 = BasicBlock_ss(cur_depth)
        self.decres2_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = channels[2]

        # Decoding stage 1
        self.decres1_1 = BasicBlock_ss(cur_depth, planes=cur_depth)
        self.decres1_2 = BasicBlock_ss(cur_depth)
        self.decres1_3 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = channels[1]

        # Decoding stage 0
        self.decres0_1 = BasicBlock_ss(cur_depth, planes=cur_depth)
        self.decres0_2 = BasicBlock_ss(cur_depth)
        self.decres0_3 = BasicBlock_ss(cur_depth)

        # Output
        # self.coupling = nn.Conv2d(cur_depth + channels[1] + CD * channels[1], label_nbr, kernel_size=1)
        self.coupling = nn.Conv2d(cur_depth, label_nbr, kernel_size=1)
        # self.sm = nn.LogSoftmax(dim=1)

    def forward(self, outputs):
        x = self.decres5_1(outputs[3])
        x = self.decres5_2(x)
        x = self.decres5_3(x)


        x = self.decres4_1(torch.cat((x, outputs[2]), 1))
        x = self.decres4_2(x)
        x = self.decres4_3(x)


        x = self.decres3_1(torch.cat((x, outputs[1]), 1))
        x = self.decres3_2(x)
        x = self.decres3_3(x)


        x = self.decres2_1(torch.cat((x, outputs[0]), 1))
        x = self.decres2_2(x)
        x = self.decres2_3(x)

        x = self.decres1_1(x)
        x = self.decres1_2(x)
        x = self.decres1_3(x)

        x = torch.nn.functional.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)

        x = self.decres0_1(x)
        x = self.decres0_2(x)
        x = self.decres0_3(x)

        x = self.coupling(x)
        # x = self.sm(x)

        return x