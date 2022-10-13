import torch.nn as nn
import torch
from torchvision import transforms
import PIL.Image as Image
import cv2

#Double convloution layer
class MultiscaleAttentionNetwork(nn.Module):
    def __init__(self, class_num):
        super(MultiscaleAttentionNetwork, self).__init__()

        #Conv1 :
        self.conv33= nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv55 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv77 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3)

        # Conv2+pooling1:
        self.conv2 = nn.Sequential(
            DoubleConv(64, 128),
            nn.MaxPool2d(2)
        )
        # Conv3:
        self.conv3 = DoubleConv(128,256)
        #Attention mechanism
        self.attention = CBAM(256)
        # Conv4+pooling:
        self.conv4 = nn.Sequential(
            DoubleConv(256, 256),
            nn.MaxPool2d(2)
        )
        # Conv5-7+pooling:
        self.conv5 = nn.Sequential(
            DoubleConv(256, 512),
            nn.MaxPool2d(2)
        )
        # Conv6+pooling:
        self.conv6 = nn.Sequential(
            DoubleConv(512, 512),
            nn.MaxPool2d(2)
        )

        # averagepool
        self.avg_pooling = nn.AdaptiveMaxPool2d((1,1))

        #Classifier
        self.classifer = nn.Linear(512,class_num)

    def forward(self, x):

       #conv1 
        x33 = self.conv33(x)
        x55 = self.conv55(x)
        x77 = self.conv77(x)
        conv1_fusion = torch.cat([x33,x55,x77],dim=1)

        x2 = self.conv2(conv1_fusion)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        avgpool = self.avg_pooling(x6)
        B, C, H, W = avgpool.size()
        avgpool = avgpool.view(B, -1)
        out = self.classifer(avgpool)
        return out

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(inplace=True)
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)

#Attention mechanism
class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # Channel-Attention
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # Spatial-Attention

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):  # Channel-Attention mechanism
    def __init__(self, in_channel, r=0.5):  # Channel is the input dimension, r is the scaling ratio of the full connection layer 
                                                                   # control the number of intermediate layers
        super(ChannelAttentionModul, self).__init__()
        # Global max-pooling
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # Global avg-pooling
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # Activ
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.Global max-pooling branch 
        max_branch = self.MaxPool(x)
        # Input MLP fcnn to obtain weight
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2. avg_branch
        avg_branch = self.AvgPool(x)
        #  avg_branch
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool  weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # Dimension splicing [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # Feature maps

        # Convolution operation results in spatial attention
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # Multiply with the original image channel
        x = Ms * x

        return x


if __name__=='__main__':

    #Definition of image preprocessing (converted to Tensor and normalized)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.496], std=[0.229, 0.224, 0.225])])

    #Read test image
    image = cv2.imread('testimage.png')

    #Image preprocessing
    img = cv2.resize(image, (128, 128))
    img = transform(img)
    img = img.unsqueeze(0)

    #design model
    net = MultiscaleAttentionNetwork(10)

    #implement
    out = net(img)

    print(out)
