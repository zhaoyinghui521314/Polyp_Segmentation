#-*- coding = utf-8 -*-
#@Time: 2021/1/23 14:42
#@Author:赵应辉
#@File:U_Net.py
#@Software:PyCharm
import torch.nn as nn
import torch




import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            #nn.InstanceNorm2d(out_ch, affine=True), # Bn -> In
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(inplace=True), # Relu -> LeakyRelu
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #ConvOffset2D(out_ch),
            nn.BatchNorm2d(out_ch),
            #nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True)
            #nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

class DenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, radio):# 输入为下采样最后一个尺度
        super(DenseBlock, self).__init__()
        self.radiof = in_ch // radio
        self.conv1 = ConvBlock(in_ch, self.radiof)
        self.conv2 = ConvBlock(in_ch + self.radiof, self.radiof)
        self.conv3 = ConvBlock(in_ch + self.radiof * 2 , out_ch)


    def forward(self, x):
        x1 = self.conv1(x)
        y = torch.cat([x, x1], dim=1)
        x2 = self.conv2(y)
        y = torch.cat([y, x2], dim=1)
        x3 = self.conv3(y)

        return x3


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x, isHotMap=False):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        x_visualize = F.interpolate(up_6, size=(352, 352), mode='bilinear', align_corners=False)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        if isHotMap:
            return c10, x_visualize
        else:
            return c10

if __name__ == '__main__':
    model = Unet(3, 1).cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    prediction1, v = model(input_tensor, isHotMap=True)
    print("input shape:", prediction1.shape, v.shape)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("params:", num_params)

    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(modela, (3, 352, 352), as_strings=True, print_per_layer_stat=True)
    # print(flops, params)

    # import thop
    # x = torch.randn(1, 3, 352, 352)
    # f, p = thop.profile(modela, inputs=(x))
    # print(f, p)
    #
    # from flopth import flopth
    # print(flopth(modela, in_size=[3, 192, 256]))



#　自适合不同尺寸
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             # nn.BatchNorm2d(out_ch),
#             # nn.InstanceNorm2d(out_ch, affine=True), # Bn -> In
#             nn.GroupNorm(32, out_ch),
#             nn.ReLU(inplace=True),
#             # nn.LeakyReLU(inplace=True), # Relu -> LeakyRelu
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             # ConvOffset2D(out_ch),
#             # nn.BatchNorm2d(out_ch),
#             # nn.InstanceNorm2d(out_ch, affine=True),
#             nn.GroupNorm(32, out_ch),
#             nn.ReLU(inplace=True),
#             # nn.LeakyReLU(inplace=True),
#
#             # nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             # nn.GroupNorm(32, out_ch),
#             # nn.ReLU(inplace=True)
#         )
#
#
#         # self.convs = nn.Sequential(
#         #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
#         #     nn.GroupNorm(32, out_ch),
#         #     nn.ReLU(inplace=True)
#         # )
#         #
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(out_ch, out_ch, 3, padding=1),
#         #     nn.GroupNorm(32, out_ch),
#         #     nn.ReLU(inplace=True)
#         # )
#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(out_ch, out_ch, 3, padding=1),
#         #     nn.GroupNorm(32, out_ch),
#         #     nn.ReLU(inplace=True)
#         # )
#         # self.conv4 = nn.Sequential(
#         #     nn.Conv2d(out_ch, out_ch, 3, padding=1),
#         #     nn.GroupNorm(32, out_ch),
#         #     nn.ReLU(inplace=True)
#         # )
#     def forward(self, input):
#         return self.conv(input)
#
#         # x1 = self.convs(input)
#         # x2 = self.conv2(x1)
#         # x3 = self.conv3(x2)
#         # x4 = self.conv4(x2 + x3)
#         # return x2 + x3 + x4
#
#
# class DenseBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, radio):  # 输入为下采样最后一个尺度
#         super(DenseBlock, self).__init__()
#         self.radiof = in_ch // radio
#         self.conv1 = ConvBlock(in_ch, self.radiof)
#         self.conv2 = ConvBlock(in_ch + self.radiof, self.radiof)
#         self.conv3 = ConvBlock(in_ch + self.radiof * 2, out_ch)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         y = torch.cat([x, x1], dim=1)
#         x2 = self.conv2(y)
#         y = torch.cat([y, x2], dim=1)
#         x3 = self.conv3(y)
#
#         return x3
#
#
# # class Unet(nn.Module):
# #     def __init__(self, in_ch, out_ch):
# #         super(Unet, self).__init__()
# #
# #         self.conv1 = DoubleConv(in_ch, 64)
# #         self.pool1 = nn.MaxPool2d(2)
# #         self.conv2 = DoubleConv(64, 128)
# #         self.pool2 = nn.MaxPool2d(2)
# #         self.conv3 = DoubleConv(128, 256)
# #         self.pool3 = nn.MaxPool2d(2)
# #         self.conv4 = DoubleConv(256, 512)
# #         self.pool4 = nn.MaxPool2d(2)
# #         self.conv5 = DoubleConv(512, 1024)
# #         self.dense = DenseBlock(1024, 1024, radio=4)
# #         self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
# #         self.lstm6 = ConvLSTM(1024, 1024, [(3, 3), (5, 5), (7, 7)], 3, True, True, False)
# #         self.conv6 = DoubleConv(1024, 512)
# #         self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
# #         self.lstm7 = ConvLSTM(512, 512, [(3, 3), (5, 5), (7, 7)], 3, True, True, False)
# #         self.conv7 = DoubleConv(512, 256)
# #         self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
# #         self.lstm8 = ConvLSTM(256, 256, [(3, 3), (5, 5), (7, 7)], 3, True, True, False)
# #         self.conv8 = DoubleConv(256, 128)
# #         self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
# #         self.lstm9 = ConvLSTM(128, 128, [(3, 3), (5, 5), (7, 7)], 3, True, True, False)
# #         self.conv9 = DoubleConv(128, 64)
# #         self.conv10 = nn.Conv2d(64, out_ch, 1)
# #         self.softmax = nn.Softmax(dim=1)
# #     def forward(self, x):
# #         c1 = self.conv1(x)
# #         print(c1.shape)
# #         p1 = self.pool1(c1)
# #         c2 = self.conv2(p1)
# #         print(c2.shape)
# #         p2 = self.pool2(c2)
# #         c3 = self.conv3(p2)
# #         p3 = self.pool3(c3)
# #         c4 = self.conv4(p3)
# #         p4 = self.pool4(c4)
# #         c5 = self.conv5(p4)
# #         d5 = self.dense(c5)
# #         up_6 = self.up6(d5)
# #         merge6 = torch.cat([up_6, c4], dim=1)
# #         merge6_LSTM, _ = self.lstm6(torch.unsqueeze(merge6, dim=0))
# #         c6 = self.conv6(merge6_LSTM[-1][0])
# #         up_7 = self.up7(c6)
# #         merge7 = torch.cat([up_7, c3], dim=1)
# #         merge7_LSTM, _ = self.lstm7(torch.unsqueeze(merge7, dim=0))
# #         c7 = self.conv7(merge7_LSTM[-1][0])
# #         up_8 = self.up8(c7)
# #         merge8 = torch.cat([up_8, c2], dim=1)
# #         merge8_LSTM, _ = self.lstm8(torch.unsqueeze(merge8, dim=0))
# #         c8 = self.conv8(merge8_LSTM[-1][0])
# #         up_9 = self.up9(c8)
# #         merge9 = torch.cat([up_9, c1], dim=1)
# #         merge9_LSTM, _ = self.lstm9(torch.unsqueeze(merge9, dim=0))
# #         c9 = self.conv9(merge9_LSTM[-1][0])
# #         c10 = self.conv10(c9)
# #         #out = nn.Sigmoid()(c10)
# #         #out = self.softmax(c10)
# #         #return c10
# #         return c10
#
# # 适合任意尺度的输入输出
# class Unet(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Unet, self).__init__()
#         self.conv1 = DoubleConv(in_ch, 64)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) #　向上取整
#         self.conv2 = DoubleConv(64, 128)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
#         self.conv3 = DoubleConv(128, 256)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
#         self.conv4 = DoubleConv(256, 512)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
#         self.conv5 = DoubleConv(512, 1024)
#         # self.dense5 = DenseBlock(1024, 1024, radio=1)
#         self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.conv6 = DoubleConv(1024, 512)
#         self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv7 = DoubleConv(512, 256)
#         self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv8 = DoubleConv(256, 128)
#         self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv9 = DoubleConv(128, 64)
#         self.conv10 = nn.Conv2d(64, out_ch, 1)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         c1 = self.conv1(x)
#         p1 = self.pool1(c1)
#         c2 = self.conv2(p1)
#         p2 = self.pool2(c2)
#         c3 = self.conv3(p2)
#         p3 = self.pool3(c3)
#         c4 = self.conv4(p3)
#         p4 = self.pool4(c4)
#         c5 = self.conv5(p4)
#         # d5 = self.dense5(c5)
#         up_6 = self.up6(c5)
#         up_6 = self.Pad(up_6, c4)
#         merge6 = torch.cat([up_6, c4], dim=1)
#         c6 = self.conv6(merge6)
#         up_7 = self.up7(c6)
#         up_7 = self.Pad(up_7, c3)
#         merge7 = torch.cat([up_7, c3], dim=1)
#         c7 = self.conv7(merge7)
#         up_8 = self.up8(c7)
#         up_8 = self.Pad(up_8, c2)
#         merge8 = torch.cat([up_8, c2], dim=1)
#         c8 = self.conv8(merge8)
#         up_9 = self.up9(c8)
#         up_9 = self.Pad(up_9, c1)
#         merge9 = torch.cat([up_9, c1], dim=1)
#         c9 = self.conv9(merge9)
#         c10 = self.conv10(c9)
#
#         return c10
#
#     def Pad(self, x, y):
#         diffX = y.shape[2] - x.shape[2]
#         diffY = y.shape[3] - x.shape[3]
#         # print(diffX, diffY)
#         x = torch.nn.functional.pad(x, (diffY, 0, diffX, 0))
#         return x
#
#
# if __name__ == '__main__':
#     # 50511299 66212419
#     input = torch.rand(1, 1, 9, 3)
#     model = Unet(1, 1)
#     output = model(input)
#     # h1.build_graph(model, input)
#     num_params = 0
#     for param in model.parameters():
#         num_params += param.numel()
#     print(num_params)
#
