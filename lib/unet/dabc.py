import torch
import torch.nn as nn
import torch.nn.functional as F


from collections import OrderedDict
import math 

class ResNetBottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, stride = 1):
        super(ResNetBottleneck,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
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
def add_stage(inplanes, outplanes, nblocks, stride_init = 1):

    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(ResNetBottleneck(
            inplanes, outplanes, stride
        ))
        inplanes = outplanes
        stride = 1
    return nn.Sequential(*res_blocks)
class ResNet50(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.block_counts = (3, 4, 6, 3)

        self.layer1 = nn.Sequential(OrderedDict([
                                    ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
                                    ('bn1', nn.BatchNorm2d(64)),
                                    ('relu', nn.ReLU(inplace=True)),
                                    ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        self.layer2 = add_stage(64, 256, self.block_counts[0], stride_init=1) 
        self.layer3 = add_stage(256, 512, self.block_counts[1], stride_init=2)
        self.layer4 = add_stage(512, 1024, self.block_counts[2], stride_init=2)
        self.layer5 = add_stage(1024, 2048, self.block_counts[3], stride_init=2)

    


    def forward(self, x):
        print("initial shape", x.shape)
        x = self.layer1(x)
        print("after layer 1: ", x.shape)
        x = self.layer2(x)
        print("after layer 2: ", x.shape)
        x = self.layer3(x)
        print("after layer 3: ", x.shape)
        x = self.layer4(x)
        print("after layer 4: ", x.shape)
        x = self.layer5(x)
        print("after layer 5: ", x.shape)

        return x

class Lateral(nn.Module):

    def __init__(self, encoder,crop_size):
        super().__init__()
        self.dim_in = [2048, 1024, 512, 256]
        self.dim_out = [512, 256, 256, 256]
        self.num_lateral_stages = 4
        self.lateral = nn.ModuleList()
        self.bottomup = encoder()
        self.crop_size = crop_size

        for i in range(self.num_lateral_stages):
            self.lateral.append(FTB_block(self.dim_in[i], self.dim_out[i]))

        self.bottomup_top = Global_pool_block(self.dim_in[0], self.dim_out[0], 16, self.crop_size)
    
    def forward(self, x):
        _, _, h, w = x.shape
        backbone_stage_size = [(math.ceil(h/(2.0**i)), math.ceil(w/(2.0**i))) for i in range(5, 1, -1)]
        backbone_stage_size.append(backbone_stage_size[-1])
        backbone_stage_size.append((h, w))
        bottomup_blocks_out = [self.bottomup.layer1(x)]
        bottomup_blocks_out.append(self.bottomup.layer2(bottomup_blocks_out[-1]))
        bottomup_blocks_out.append(self.bottomup.layer3(bottomup_blocks_out[-1]))
        bottomup_blocks_out.append(self.bottomup.layer4(bottomup_blocks_out[-1]))
        bottomup_blocks_out.append(self.bottomup.layer5(bottomup_blocks_out[-1]))
        # for i in range(0, self.bottomup.convX):
        #     bottemup_blocks_out.append(
        #         getattr(self.bottomup, 'res%d' % (i + 1))(bottemup_blocks_out[-1])
        #     )

        bottomup_top_out = self.bottomup_top(bottomup_blocks_out[-1])
        lateral_blocks_out = [bottomup_top_out]
        for i in range(self.num_lateral_stages):
            lateral_blocks_out.append(self.lateral[i](
                bottomup_blocks_out[-(i + 1)]
            ))
        return lateral_blocks_out, backbone_stage_size


class FCN_topdown_block(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.afa_block = AFA_block(dim_in)
        self.ftb_block = FTB_block(self.dim_in, self.dim_out)

    def forward(self, lateral, top, size=None):
        if lateral.shape != top.shape:
            h, w = lateral.size(2), lateral.size(3)
            top = F.interpolate(input=top, size=(h, w), mode='bilinear',align_corners=True)
        out = self.afa_block(lateral, top)
        out = self.ftb_block(out)
        return out

class FCN_topdown_predict(nn.Module):
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

class FCN_last_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ftb = FTB_block(dim_in, dim_out)

    def forward(self, input, backbone_stage_size):
        out = F.upsample(input=input, size=(backbone_stage_size[4][0], backbone_stage_size[4][1]), mode='bilinear', align_corners=True)
        out = self.ftb(out)
        out = F.upsample(input=out, size=(backbone_stage_size[5][0], backbone_stage_size[5][1]), mode='bilinear', align_corners=True)
        return out

class FCN_topdown(nn.Module):

    def __init__(self):
        super().__init__()

        self.dim_in = [512, 256, 256, 256, 256, 256]
        self.dim_out = [256, 256, 256, 256, 256, 150]
        self.num_fcn_topdown = len(self.dim_in)
        self.top = nn.Sequential(
            nn.Conv2d(self.dim_in[0] , self.dim_in[0], 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.dim_in[0], 0.5)
        )
        self.topdown_fcn1 = FCN_topdown_block(self.dim_in[0], self.dim_out[0])
        self.topdown_fcn2 = FCN_topdown_block(self.dim_in[1], self.dim_out[1])
        self.topdown_fcn3 = FCN_topdown_block(self.dim_in[2], self.dim_out[2])
        self.topdown_fcn4 = FCN_topdown_block(self.dim_in[3], self.dim_out[3])
        self.topdown_fcn5 = FCN_last_block(self.dim_in[4], self.dim_out[4])
        self.topdown_predict = FCN_topdown_predict(self.dim_in[5], self.dim_out[5])
    
    def forward(self, laterals, backbone_stage_size):
        x = self.top(laterals[0])
        x1 = self.topdown_fcn1(laterals[1], x)
        x2 = self.topdown_fcn2(laterals[2], x1)
        x3 = self.topdown_fcn3(laterals[3], x2)
        x4 = self.topdown_fcn4(laterals[4], x3)
        x5 = self.topdown_fcn5(x4, backbone_stage_size)
        x6 = self.topdown_predict(x5)
        return x6

class DepthModel(nn.Module):
    def __init__(self, crop_size):
        super(DepthModel, self).__init__()
        self.encoder_modules = Lateral(ResNet50, crop_size)
        self.decoder_modules = FCN_topdown()

    def forward(self, x):
        lateral_out, encoder_stage_size = self.encoder_modules(x)
        out_logit, out_softmax = self.decoder_modules(lateral_out, encoder_stage_size)
        return out_logit, out_softmax
       
class Global_pool_block(nn.Module):
    def __init__(self, dim_in, dim_out, output_stride, crop_size):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.globalpool_conv1x1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.globalpool_bn = nn.BatchNorm2d(self.dim_out, momentum=0.9)
        self.unpool = nn.AdaptiveAvgPool2d((int(crop_size[0] / output_stride), int(crop_size[1] / output_stride)))

    def forward(self, x):
        out = self.globalpool_conv1x1(x)
        out = self.globalpool_bn(out)
        out = self.globalpool(out)
        out = self.unpool(out)
        return out


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