import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vgg import VGG16
import matplotlib.pyplot as plt
from model.FRM import FRM
from model.resnet import resnet50

def huatu(filter, name):
    plt.figure(figsize=(20, 10))
    layer_viz = filter[0, :, :, :]
    for i, filter in enumerate(layer_viz):
        if i >= 18:
            break
        plt.subplot(3, 6, i + 1)
        plt.imshow(filter.cpu().detach().numpy())
        plt.axis('off')
    plt.suptitle(name, fontsize=56)
    plt.legend()
    plt.savefig(name + str('.svg'))
    plt.close()

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

class MSTLoc(nn.Module):
    def __init__(self):
        super(MSTLoc, self).__init__()
        self.vgg = VGG16()
        self.resnet = resnet50()
        self.m1 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.m2 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.m3 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.rfb2_1 = RFB_modified(256, 64)
        self.rfb3_1 = RFB_modified(512, 64)
        self.rfb4_1 = RFB_modified(512, 64)
        self.edge_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.edge_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.ra4_conv2 = BasicConv2d(128, 128, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(128, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)

        self.ra3_conv2 = BasicConv2d(128, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.ra2_conv2 = BasicConv2d(128, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 32, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(32, 16, kernel_size=3, padding=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.adjust_channels = BasicConv2d(336, 1024, kernel_size=1)

        self.mlp_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.vgg.conv1(x)  # 112
        x1 = self.vgg.conv2(x)  # 56
        x2 = self.vgg.conv3(x1)  # 28
        x3 = self.vgg.conv4_1(x2)  # 14
        x4 = self.vgg.conv5_2(x3)  # 7
        x2 = self.rfb2_1(x2)
        x3 = self.rfb3_1(x3)
        x4 = self.rfb4_1(x4)
        x = self.edge_conv1(x1)
        x = self.edge_conv2(x)
        edge_guidance = self.edge_conv3(x)  # 176
        x_t1 = torch.cat((x2, F.interpolate(edge_guidance, scale_factor=0.5, mode='bilinear', align_corners=True)), dim=1)
        x_t1 = F.relu(self.ra2_conv2(x_t1))
        x_t1 = F.relu(self.ra2_conv3(x_t1))
        x_t1 = F.relu(self.ra2_conv4(x_t1))

        x_t2 = torch.cat((x3, F.interpolate(edge_guidance, scale_factor=0.25, mode='bilinear', align_corners=True)), dim=1)
        x_t2 = F.relu(self.ra3_conv2(x_t2))
        x_t2 = F.relu(self.ra3_conv3(x_t2))
        x_t2 = F.relu(self.ra3_conv4(x_t2))

        x_t3 = torch.cat((x4, F.interpolate(edge_guidance, scale_factor=0.125, mode='bilinear', align_corners=True)), dim=1)
        x_t3 = F.relu(self.ra4_conv2(x_t3))
        x_t3 = F.relu(self.ra4_conv3(x_t3))
        x_t3 = F.relu(self.ra4_conv4(x_t3))

        x_t3_upsampled = F.interpolate(x_t3, scale_factor=4, mode='bilinear', align_corners=True)
        x_t2_upsampled = F.interpolate(x_t2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x_t1, x_t2_upsampled, x_t3_upsampled), dim=1)

        x = self.global_pool(x)
        x = self.adjust_channels(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_head(x)
        x = F.sigmoid(x)
        return x

if __name__ == "__main__":
    model = MSTLoc()
    img = torch.rand(size=(1, 3, 512, 512))
    model.eval()
    with torch.no_grad():
        output = model(img)
    print(output)