import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

'''========== ChannelAttention =========='''
class CA(nn.Module):
    def __init__(self, in_planes):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

'''========== SpatialAttention =========='''
class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()

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

class FESM_attention(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FESM_attention, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = CA(planes)
        self.sa = SA()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(out)
        # The output of ca is the weights calculated by channel attention
        # Calculate importance, positive factor
        ca_important = self.ca(out)
        # Calculate the number of pixels to select
        ca_num = ca_important.numel()
        # Calculate the top 95 percent
        top_k1 = int(ca_num * 0.95)
        # Set threshold
        ca_threshold = torch.topk(ca_important.view(-1),top_k1).values.min()
        # Generate mask
        ca_mask = (ca_important >= ca_threshold).float()
#        print(ca_mask)
        # Apply mask, generate suppression factor
        ca_important_suppression = 1 - ca_mask

        ca_out_salient = out + ca_important * out
        ca_out_suppression = out * ca_important_suppression
        # Calculate spatial attention
        # Calculate importance, positive factor
        sa_important = self.sa(out)
        # Calculate the number of pixels to select
        sa_num = sa_important.numel()
        # Calculate the top 95 percent
        top_k1 = int(sa_num * 0.95)
        # Set threshold
        sa_threshold = torch.topk(sa_important.view(-1),top_k1).values.min()
        # Generate mask
        sa_mask = (sa_important >= sa_threshold).float()
        # Apply mask, generate suppression factor
        sa_important_suppression = 1-sa_mask



        sa_out_salient = out + sa_important * out
        sa_out_suppression = out * sa_important_suppression

        out_salient = ca_out_salient + sa_out_salient
        out_suppression = ca_out_suppression + sa_out_suppression

        return out_salient,out_suppression


if __name__ == "__main__":
    model = FESM_attention(256,256)
    img = torch.rand(size=(1,256,256,256))
    out_salient = model(img)[0]
    out_suppression = model(img)[1]
    print(out_salient)
    print(out_suppression)