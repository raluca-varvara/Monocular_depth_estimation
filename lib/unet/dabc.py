import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, stride = 1):
        super(ResNetBottleneck,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilate, dilation=dilate, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module('conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet50(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.block_counts = (3, 4, 6, 3)

        self.layer1 = nn.Sequential(OrderedDict([
                                    ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
                                    ('bn1', nn.BatchNorm2d(64)),
                                    ('relu', nn.ReLU(inplace=True)),
                                    ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        self.layer2 = add_stage(64, 256, block_counts[0], stride_init=1) 
        self.layer3 = add_stage(256, 512, block_counts[1], stride_init=2)
        self.res4 = add_stage(512, 1024, block_counts[2], stride_init=res4_stride)
        self.layer4 = add_stage(1024, 2048, block_counts[3], stride_init=res5_stride)

    
    def add_stage(inplanes, outplanes, nblocks, stride_init=2):

        res_blocks = []
        stride = stride_init
        for _ in range(nblocks):
            res_blocks.append(ResNeXtBottleneck(
                inplanes, outplanes, stride, dilation, cardinality, base_width
            ))
            inplanes = outplanes
            stride = 1
        return nn.Sequential(*res_blocks)



class AFA_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_channels = dim * 2
        self.out_channels = dim
        self.mid_channels = int(dim / 8)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(self.in_channels, self.mid_channels, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.mid_channels, self.out_channels, 1, stride=1, padding=0, bias=False)
        self.sigmd = nn.Sigmoid()

    def forward(self, lateral, top):
        w = torch.cat([lateral, top], 1)
        w = self.globalpool(w)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w)
        w = self.sigmd(w)
        out = w * lateral + top
        return out
    
class FTB_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=2, dilation=2, bias=True)
        self.bn1 = nn.BatchNorm2d(self.out_channels, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=2, dilation=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        out = self.relu(out)
        return out

class FCN_last_block_predict(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = nn.Dropout2d(0.0)
        self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x_softmax = self.softmax(x)
        return x, x_softmax