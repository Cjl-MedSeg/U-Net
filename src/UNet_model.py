from mindspore import nn
import mindspore.numpy as np
import mindspore.ops as ops
import mindspore.ops.operations as F

def double_conv(in_ch, out_ch):
    return nn.SequentialCell(nn.Conv2d(in_ch, out_ch, 3),
                              nn.BatchNorm2d(out_ch), nn.ReLU(),
                              nn.Conv2d(out_ch, out_ch, 3),
                              nn.BatchNorm2d(out_ch), nn.ReLU())
class UNet(nn.Cell):
    def __init__(self, in_ch = 3, n_classes = 1):
        super(UNet, self).__init__()
        self.concat1 = F.Concat(axis=1)
        self.concat2 = F.Concat(axis=1)
        self.concat3 = F.Concat(axis=1)
        self.concat4 = F.Concat(axis=1)
        self.double_conv1 = double_conv(in_ch, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv2 = double_conv(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv3 = double_conv(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv4 = double_conv(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv5 = double_conv(512, 1024)

        self.upsample1 = nn.ResizeBilinear()
        self.double_conv6 = double_conv(1024 + 512, 512)
        self.upsample2 = nn.ResizeBilinear()
        self.double_conv7 = double_conv(512 + 256, 256)
        self.upsample3 = nn.ResizeBilinear()
        self.double_conv8 = double_conv(256 + 128, 128)
        self.upsample4 = nn.ResizeBilinear()
        self.double_conv9 = double_conv(128 + 64, 64)

        self.final = nn.Conv2d(64, n_classes, 1)
        # self.sigmoid = ops.Sigmoid()

    def construct(self, x):

        feature1 = self.double_conv1(x)
        tmp = self.maxpool1(feature1)
        feature2 = self.double_conv2(tmp)
        tmp = self.maxpool2(feature2)
        feature3 = self.double_conv3(tmp)
        tmp = self.maxpool3(feature3)
        feature4 = self.double_conv4(tmp)
        tmp = self.maxpool4(feature4)
        feature5 = self.double_conv5(tmp)

        up_feature1 = self.upsample1(feature5, scale_factor=2)
        tmp = self.concat1((feature4, up_feature1))
        tmp = self.double_conv6(tmp)
        up_feature2 = self.upsample2(tmp, scale_factor=2)
        tmp = self.concat2((feature3, up_feature2))
        tmp = self.double_conv7(tmp)
        up_feature3 = self.upsample3(tmp, scale_factor=2)
        tmp = self.concat3((feature2, up_feature3))
        tmp = self.double_conv8(tmp)
        up_feature4 = self.upsample4(tmp, scale_factor=2)
        tmp = self.concat4((feature1, up_feature4))
        tmp = self.double_conv9(tmp)
        output = self.final(tmp)

        return output