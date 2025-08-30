import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vgg import VGG16
from model.vit import ViT
import matplotlib.pyplot as plt
from model.FRM import FRM
from model.FESM_attention_95 import FESM_attention
from model.resnet import resnet50


def huatu(filter, name):
    plt.figure(figsize=(20, 10))
    layer_viz = filter[0, :, :, :]
    for i, filter in enumerate(layer_viz):
        if i >= 18:
            break
        plt.subplot(3, 6, i + 1)
        # plt.imshow(filter.cpu().detach().numpy(),cmap='binary')
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


class MSTLoc_attention(nn.Module):
    def __init__(self):
        super(MSTLoc_attention, self).__init__()
        self.vgg = VGG16()
        self.resnet = resnet50()
        self.m1 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.m2 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.m3 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.vit1 = ViT(image_size=64, patch_size=8, channels=336, dim=1024, depth=6, heads=8, mlp_dim=1024,
                        dropout=0.1, emb_dropout=0.1)
        self.vit2 = ViT(image_size=64, patch_size=8, channels=336, dim=1024, depth=6, heads=8, mlp_dim=1024,
                        dropout=0.1, emb_dropout=0.1)
        self.vit3 = ViT(image_size=64, patch_size=8, channels=336, dim=1024, depth=6, heads=8, mlp_dim=1024,
                        dropout=0.1, emb_dropout=0.1)
        self.vit4 = ViT(image_size=64, patch_size=8, channels=336, dim=1024, depth=6, heads=8, mlp_dim=1024,
                        dropout=0.1, emb_dropout=0.1)
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

#        self.frm = FRM(336, 336)
        self.FESM1 = FESM_attention(336, 336)
        self.FESM2 = FESM_attention(336, 336)
        self.FESM3 = FESM_attention(336, 336)
        self.FESM4 = FESM_attention(336, 336)



        self.mlp_head1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 6)
        )

        self.mlp_head2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 6)
        )
        self.mlp_head3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 6)
        )
        self.mlp_head4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 6)
        )

        x = nn.Upsample()


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
        # print(x_t1.shape)
        # The following are features after passing through vit
        # x_t1 = self.vit1(x_t1)

        x_t2 = torch.cat((x3, F.interpolate(edge_guidance, scale_factor=0.25, mode='bilinear', align_corners=True)), dim=1)
        x_t2 = F.relu(self.ra3_conv2(x_t2))
        x_t2 = F.relu(self.ra3_conv3(x_t2))
        x_t2 = F.relu(self.ra3_conv4(x_t2))

        # x_t2 = self.vit2(x_t2)
        x_t3 = torch.cat((x4, F.interpolate(edge_guidance, scale_factor=0.125, mode='bilinear', align_corners=True)), dim=1)
        x_t3 = F.relu(self.ra4_conv2(x_t3))
        x_t3 = F.relu(self.ra4_conv3(x_t3))
        x_t3 = F.relu(self.ra4_conv4(x_t3))

        # x_t3 = self.vit3(x_t3)
        x_t3_upsampled = F.interpolate(x_t3, scale_factor=4, mode='bilinear', align_corners=True)
        x_t2_upsampled = F.interpolate(x_t2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x_t1, x_t2_upsampled, x_t3_upsampled), dim=1)
#        For outputting FESM1
        features_before_FESM1 = x.clone()

        # Start classification

        x_enhance = self.FESM1(x)[0]
        x_enhance = self.vit1(x_enhance)
        x1 = x_enhance.view(x_enhance.size(0), -1)
        x1 = self.mlp_head1(x1)
        # Get the first probability
        #        x1 = F.sigmoid(x1)
        # First layer output, second layer input
        x_suppression = self.FESM1(x)[1]
        features_after_FESM1 = x_suppression.clone()
        # Second layer
        x_enhance = self.FESM2(x_suppression)[0]
        x_enhance = self.vit2(x_enhance)
        x2 = x_enhance.view(x_enhance.size(0), -1)
        x2 = self.mlp_head2(x2)
        # Get the second probability
        #        x2 = F.sigmoid(x2)
        # Second layer output, third layer input
        x_suppression = self.FESM2(x_suppression)[1]
        features_after_FESM2 = x_suppression.clone()
        # Third layer
        x_enhance = self.FESM3(x_suppression)[0]
        x_enhance = self.vit3(x_enhance)
        x3 = x_enhance.view(x_enhance.size(0), -1)
        x3 = self.mlp_head3(x3)
        # Get the third probability
        #        x3 = F.sigmoid(x3)
        # Third layer output, fourth layer input
        x_suppression = self.FESM3(x_suppression)[1]
        features_after_FESM3 = x_suppression.clone()
        # Fourth layer
        x_enhance = self.FESM4(x_suppression)[0]
        x_enhance = self.vit4(x_enhance)
        x4 = x_enhance.view(x_enhance.size(0), -1)
        x4 = self.mlp_head4(x4)
        # Get the fourth probability
        #        x4 = F.sigmoid(x4)

        x = x1 + x2 + x3+x4
        x = F.sigmoid(x)

#        return x,features_before_FESM1,features_after_FESM1,features_after_FESM2,features_after_FESM3


        return x

if __name__ == "__main__":
    model = MSTLoc_attention()
    img = torch.rand(size=(1, 3, 256, 256))
    print(model(img))