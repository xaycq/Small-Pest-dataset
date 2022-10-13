from torchvision import transforms
import torch.nn as nn
from torch.nn import functional as F
import torch

class ThreeParallelCNN(nn.Module):
    def __init__(self, class_num):
        super(ThreeParallelCNN, self).__init__()

        self.CNN1 = RestNet18(class_num)
        self.CNN2 = RestNet18(class_num)
        self.CNN3 = RestNet18(class_num)

        self.pool2 = nn.MaxPool2d(kernel_size=2,ceil_mode=True)

        self.classifer = nn.AdaptiveAvgPool1d(0)

    def forward(self, input):
        x = self.CNN1(input)

        x_down2= self.pool2(input)
        x2 = self.CNN2(x_down2)

        x_down4 = self.pool2(x_down2)
        x3 = self.CNN3(x_down4)

        out = torch.cat([x,x2,x3],dim=0)
        output = torch.mean(out,dim=0)
        output = output.unsqueeze(0)
        return output


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self, class_num):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, class_num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out




if __name__=='__main__':

    #图像预处理定义（转为Tensor并归一化）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.496], std=[0.229, 0.224, 0.225])])

    net  = ThreeParallelCNN(10)
    rand = torch.rand(1,3,128,128)
    out = net(rand)
    # #读取测试图像
    # image = cv2.imread('testimage.png')
    #
    # #图像预处理
    # img = cv2.resize(image, (128, 128))
    # img = transform(img)
    # img = img.unsqueeze(0)
    #
    # #定义模型
    # net = MultiscaleAttentionNetwork(10)
    #
    # #运行
    # out = net(img)

    print(out)
