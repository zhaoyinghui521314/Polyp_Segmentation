# 开始消融：
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.segformer.segformer import mit_b2, mit_b3, mit_b4
# from segformer.segformer import mit_b2, mit_b3, mit_b4
from backbone.cswin.cswin import CSWin
from backbone.p2t.p2t import p2t_tiny
from backbone.pvt.pvt_v2 import pvt_v2_b2, pvt_v2_b3
from backbone.rest.rest import rest_base, rest_large
from backbone.segformer.segformer import mit_b2, mit_b3, mit_b4
from backbone.swin.swin import SwinTransformer
from backbone.twins.twins import pcpvt_base_v0, pcpvt_large
import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(16, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 变化到相同维度可以cat
        # print(x_h.shape, x_w.shape) # n, c, h, 1
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)  # n, 8, 2 * h, 1
        # print(y.shape)
        x_h, x_w = torch.split(y, [h, w], dim=2)  # 分开
        x_w = x_w.permute(0, 1, 3, 2)
        # print(x_w.shape, x_h.shape)# n, 8, 1, h    n, 8, h, 1
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # print(a_w.shape, a_h.shape)# n, 3, 1, h   n, 3, h, 1
        # print(identity.shape)# n, c, h, w
        out = identity * a_w * a_h

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3, g):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 + x2 * g
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(
            self.upsample(x2)) * x3 + x3 * g

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class sMLPBlock(nn.Module):
    def __init__(self, h=44, w=44, c=64):
        super().__init__()
        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)
        self.fuse = nn.Linear(3 * c, c)

    def forward(self, x):
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id], dim=1)
        out = self.fuse(x_fuse.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class gap(nn.Module):
    def __init__(self):
        super(gap, self).__init__()
        self.glob = nn.AdaptiveAvgPool2d(1)
        self.glob_c1 = nn.Conv2d(64, 16, kernel_size=1)
        self.glob_c2 = nn.Conv2d(16, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.glob(x)
        x = self.glob_c2(self.relu(self.glob_c1(x)))
        return self.sigmoid(x)


class SpatialAttention2(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention2, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x1 + x2 + x3
        return self.sigmoid(x)


class CFMN(nn.Module):
    def __init__(self, channel):
        super(CFMN, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv3 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1  # 1
        o1 = self.conv2(x1)
        # print("shape1", o1.shape)
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)  # 2
        o2 = self.conv3(x2_2)
        # print("shape2", o2.shape)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        o3 = self.conv4(x3_2)
        # print("shape3", o3.shape)
        return [o1, o2, o3]


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNetEncoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = Down(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 320)
        self.down4 = Down(320, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class backbone(nn.Module):
    def __init__(self, mode='segformer', isLarge=False):
        super(backbone, self).__init__()
        print("Load Backbone...!")
        if mode == 'pvtv2':
            if isLarge:
                self.backbone = pvt_v2_b3()  # [64, 128, 320, 512]
                path = r'E:\Polyp_Segmentation\backbone\pvt\pvt_v2_b3.pth'
            else:
                self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
                path = r'E:\Polyp_Segmentation\backbone\pvt\pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'cswin':
            if isLarge:
                self.backbone = CSWin(embed_dim=96,
                                      depth=[2, 4, 32, 2],
                                      num_heads=[4, 8, 16, 32],
                                      split_size=[1, 2, 7, 7])
                path = r'E:\Polyp_Segmentation\backbone\cswin\cswin_base_384.pth'
            else:
                self.backbone = CSWin(embed_dim=96,
                                      depth=[2, 4, 32, 2],
                                      num_heads=[4, 8, 16, 32],
                                      split_size=[1, 2, 7, 7])
                path = r'E:\Polyp_Segmentation\backbone\cswin\cswin_base_224.pth'
            save_model = torch.load(path)['state_dict_ema']
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [96, 192, 384, 768]
        elif mode == 'p2t':
            self.backbone = p2t_tiny().cuda()
            path = r'E:\Polyp_Segmentation\backbone\p2t\p2t_tiny.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'rest':
            if isLarge:
                self.backbone = rest_large().cuda()
                path = r'E:\Polyp_Segmentation\backbone\rest\rest_large.pth'
            else:
                self.backbone = rest_base().cuda()
                path = r'E:\Polyp_Segmentation\backbone\rest\base.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [96, 192, 384, 768]
        elif mode == 'segformer':
            print("segformer...!")
            if isLarge:
                self.backbone = mit_b3().cuda()
                path = r'E:\Polyp_Segmentation\backbone\segformer\mit_b3.pth'
            else:
                self.backbone = mit_b2().cuda()
                path = r'E:\Polyp_Segmentation\backbone\segformer\mit_b2.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'swin':
            if isLarge:
                self.backbone = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2],
                                                num_heads=[4, 8, 16, 32], window_size=7)
                path = r'E:\Polyp_Segmentation\backbone\Swin\swin_base_patch4_window7_384.pth'
            else:
                self.backbone = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2],
                                                num_heads=[4, 8, 16, 32], window_size=7)
                path = r'E:\Polyp_Segmentation\backbone\Swin\swin_base_patch4_window7_224.pth'
            save_model = torch.load(path)['model']
            for key in save_model:
                print(key)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [128, 256, 512, 1024]
        elif mode == 'twins':
            self.backbone = pcpvt_large()
            path = r'E:\Polyp_Segmentation\backbone\twins\pcpvt_large.pth'
            save_model = torch.load(path)
            # for key in save_model:
            #     print(key)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # 版本1:上下两条支路，下面保持原来不变，为级联上采样+空间注意力,上面融合1为相加，融合2也为相加，融合2的尺寸按照小的，上采样八倍
        print("state:", state_dict)
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        channel = 64
        # print(state_dict)
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # 1. 添加Unet Ecoder对灰度图编码得到不同层级的CNN特征图
        self.unetEncoder = UNetEncoder(1).cuda()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # self.sa3 = SpatialAttention()
        # self.sa4 = SpatialAttention()
        # self.sa5 = SpatialAttention()
        # self.ca3 = ChannelAttention(128)
        # self.ca4 = ChannelAttention(320)
        # self.ca5 = ChannelAttention(512)
        # 2. 添加轴向注意力增强不同层级的CNN特征图
        # 消融FA
        # self.coord2 = CoordAtt(64, 64)
        self.coord3 = CoordAtt(128, 128)
        self.coord4 = CoordAtt(320, 320)
        self.coord5 = CoordAtt(512, 512)
        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        # self.CFM = CFM(channel)
        # 3. 增加编码多分支输出深监督并添加边缘损失
        self.CFMN = CFMN(channel)
        self.convo1 = nn.Conv2d(channel, 1, 1)
        self.convo2 = nn.Conv2d(channel, 1, 1)
        self.convo3 = nn.Conv2d(channel, 1, 1)
        self.conv_out1 = nn.Conv2d(channel, 1, 1)
        self.conv_out2 = nn.Conv2d(channel, 1, 1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        # self.glob = gap()
        self.confuse1 = BasicConv2d(64 * 2, channel, 1)
        self.rbf = RFB_modified(64, 64)
        # 消融RFB
        # self.rbf2 = RFB_modified(128, 128)
        # self.rbf3 = RFB_modified(320, 320)
        # self.rbf4 = RFB_modified(512, 512)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, x, y, isHotMap=False):
        # backbone:
        # seg_former:
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        p1, p2, p3, p4, p5 = self.unetEncoder(y)
        x_f = p2
        x_f = self.rbf(x_f)
        p3 = self.coord3(p3)
        p4 = self.coord4(p4)
        p5 = self.coord5(p5)
        x2 = x2 + p3
        x3 = x3 + p4
        x4 = x4 + p5
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        o1, o2, o3 = self.CFMN(x4_t, x3_t, x2_t)

        oo1 = self.convo1(o1)
        oo2 = self.convo2(o2)
        oo3 = self.convo3(o3)
        x_visualize = F.interpolate(oo3, size=(352, 352), mode='bilinear', align_corners=False)
        # 1*1卷积之后 直接16, 8, 4倍上采样！！！
        prediction_32 = F.interpolate(oo1, scale_factor=32, mode='bilinear')
        prediction_16 = F.interpolate(oo2, scale_factor=16, mode='bilinear')
        prediction_8 = F.interpolate(oo3, scale_factor=8, mode='bilinear')
        x_f = self.Translayer2_0(x_f)
        x_f = self.down(x_f)
        x_f = x_f + o3
        x_f = x_f * self.ca(x_f)
        x_f = self.conv_out2(x_f)
        prediction2_8 = F.interpolate(x_f, scale_factor=8, mode='bilinear')
        if isHotMap:
            return prediction_32, prediction_16, prediction_8, prediction2_8, x_visualize
        else:
            return prediction_32, prediction_16, prediction_8, prediction2_8


if __name__ == '__main__':
    model = backbone().cuda()
    input_tensor = torch.randn(4, 3, 352, 352).cuda()
    input_tensor2 = torch.randn(4, 1, 352, 352).cuda()
    prediction1, prediction2, prediction3, prediction4, x = model(input_tensor, input_tensor2, isHotMap=True)
    print(prediction1.shape, prediction2.shape, prediction3.shape, prediction4.shape, x.shape)
    # print(model)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)

