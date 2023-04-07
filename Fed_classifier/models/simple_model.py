import torch
import torch.nn as nn
import torch.nn.functional as F

# img_shape = (opt.channels, opt.img_size, opt.img_size)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# 核心思想：
# 将原始的 MLP 组成的 GAN ，变为 CNN
# 在生成器中，将 噪声 100 * 1 * 1，和独热的标签 10 * 1 * 1 的独热标签经过一个转置卷积后 拼接，然后继续通过转置卷积
# 在判别器中，将 图片 1 * 32 * 32，和标签生成的图片 10 * 32 * 32 （其中一个通道的全为1 ，其他通道的值全为0），拼接，再通过CNN进行判别


# G(z)
class Generator( nn.Module ):     # 转置卷积输出形状的计算， 需要分情况
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        # 将 1x1 变为 4 x 4
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)  # 2D转置卷积算子。  ？？，上采样
                                                                # 参数和卷积相同
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)   # 保证形状相同，后面进行连接
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)

        # 下面的转置卷积都是将原先的输出尺寸 x 2
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)  # 核大小 4， stride = 2， padding =1可以将图像大小翻倍
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        # 最终输出尺寸为 32

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))    # 随机噪声
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)    # 通道维度拼接
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d // 2, 4, 2, 1)    # 输出形状的计算
        self.conv1_2 = nn.Conv2d(10, d // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)  # 感觉也可以先拼接，再操作
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x

# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),    # 吸收
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 10),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)
#
#         return validity

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.l1 = torch.nn.Linear(784, 512)
        self.l1 = torch.nn.Linear(1024, 512)    # 32 * 32
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
    def forward(self, x):       # pytorh如何实现函数定义时是一个x， 但是却可以输入一个批量的值 ？？？？？
        # x = x.view(-1, 784)
        x = x.view(-1, 1024)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels,channels,kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels,channels,kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        return F.relu(x + y)

class ResModel(torch.nn.Module):
    def __init__(self, in_ch =1):
        super(ResModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_ch,16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(32)
        self.conv3 = torch.nn.Conv2d(32,16,kernel_size=3)
        self.fc = torch.nn.Linear(144,10)
        self.bn1= nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        # 初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def forward(self,x):
        batch = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.mp(x)
        x = self.block1(x)
        # self.bn1(x) # BN加快收敛
        x = self.bn1(x)  # BN加快收敛
        x = F.relu(self.conv2(x))
        x = self.mp(x)
        x = self.block2(x)
        # self.bn2(x) # BN加快收敛
        x = self.bn2(x)  # BN加快收敛
        x = F.relu(self.conv3(x))
        x = x.view(batch, -1)
        x = self.fc(x)
        return x

# 适用于 MNIST和 FashionMNIST ，图片 28x 28, 单通道
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5)     # torch 的结构和基础 ，nn模块
        self.bn1 = nn.BatchNorm2d(10) # 通道数 , BN 层可以加速收敛
        self.pooling = torch.nn.MaxPool2d(2) #高宽减半
        self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)  # 通道数
        self.conv3 = torch.nn.Conv2d(in_channels=20,out_channels=20,kernel_size=3)
        self.fc = torch.nn.Linear(in_features= 180,out_features=10)        # 320 = 20 x 4x 4

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = self.bn1(x)
        x = F.relu(self.pooling(self.conv2(x)))
        # x = self.bn2(x)
        # x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1) # flatten
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # 测试输出维度
    # x = torch.randn(10,1,28,28)
    x = torch.randn(10,1,32,32)
    # model = ResModel()
    model = CNN()
    y = model(x)
    print(y.size())