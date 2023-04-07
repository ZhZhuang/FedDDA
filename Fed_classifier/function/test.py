import random
import numpy
from torch import nn
import torch
# import torch.utils.
import torch.nn.functional as F     # torch 包的结构？
                                    # torch 的架构
                                    # torch 的实现机理


# a = 1,
# b = 1
# print(type(a),a)
# print(type(b))


# L = [1,2,3,4,5]
# # t = random.sample(L, 10)
# # print(t)
# t = numpy.random.choice(L, size=10, replace=True)
# print(t, type(t))


##########  大问题 ：忘了cDCGAN的前几层是包含了 label的判别
#########   解决： 暂定使用类似于U-net的特征融合 结合  生辰器
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
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)   # nn.Conv和 nn.functional中的区别

        # self.classifier = nn.Sequential(nn.Conv2d(d*4, d*2, 3),
        #                                 nn.AdaptiveAvgPool2d(1), # Adaptive 自适应的输出1
        #                                 nn.Linear(256, 10)
        #                                 )

        self.clr_conv1 = nn.Conv2d(d*4, d*2, 3)
        self.clr_conv1_bn = nn.BatchNorm2d(d*2)     # bn 层有参数， ！！！！！ 原理
        self.clr_linear = nn.Linear(256,10)

        # F.adaptive_avg_pool2d(out, 1)     # 自适应平均池化层

    # # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label , claaifier = False):
        batch= input.size(0)    # 获取批量大小
        x = F.leaky_relu(self.conv1_1(input), 0.2)  # 感觉也可以先拼接，再操作
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)     # -> 512 * 4 * 4
        if not claaifier:    # 判别是否属于 真实样本
            x = F.sigmoid(self.conv4(x))
        else:                # 分类
            x = F.sigmoid(self.clr_conv1_bn(self.conv4(x)))
            x = F.adaptive_avg_pool2d(x, 1)  # 自适应池化层
            x = x.view(batch,-1)
            x = self.clr_linear(x)
        return x

x = torch.randn(1,1,32,32)
y = torch.randn((1,10,32,32))

model = Discriminator()
z = model(x, y)
print(z)