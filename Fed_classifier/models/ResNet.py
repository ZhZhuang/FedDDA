# import packages
import random

import numpy as np
import torch
import torchvision
from torch.backends import cudnn
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
from utils import get_logger



# Define 3x3 convolution.
def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# Define Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Define ResNet-18
class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes, in_ch = 3):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(in_ch, 16)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)
        self.layer1 = self._make_layers(block, 16, layers[0])
        self.layer2 = self._make_layers(block, 32, layers[1], 2)
        self.layer3 = self._make_layers(block, 64, layers[2], 2)
        self.layer4 = self._make_layers(block, 128, layers[3], 2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(128, num_classes)

    def _make_layers(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = torch.nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                torch.nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




# For updating learning rate.
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 此时模型
def test(model , test_loader):
    # model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda*()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return correct / total

# 训练模型
def train(model, train_loader, num_epochs= 80,
          print_fn= None,
          scheduler= None,
          curr_lr= 0.001,
          test_loader= None):
    if print_fn == None:
        print_fn = print

    top_acc = 0.0
    top_epoch = 0
    acc_list = []

    steps = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # images = images.to(device)
            # labels = labels.to(device)
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass.
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()   # 放的位置有影响？
            loss.backward()
            optimizer.step()


            if (i + 1) % 100 == 0:
                print_fn('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                            loss.item()))

            # if scheduler != None:
            #     scheduler.step()
            #     # print(optimizer.param_groups[0]["lr"])

        # # Decay learning rate.
        # if (epoch + 1) % 20 == 0:
        #     curr_lr /= 3
        #     update_lr(optimizer, curr_lr)


            steps += 1

        # Test the model.
        acc = test(model, test_loader=test_loader)
        acc_list.append(acc)
        print_fn('Accuracy: {} %, at {} round.'.format(100 * acc, epoch))
        if acc > top_acc:
            top_acc = acc
            top_epoch = epoch

    print_fn(acc_list)
    print_fn("top acc: {}, at {} epoch".format(top_acc, top_epoch))

def init( seed = 1234 ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True  # 随机数种子seed确定时，模型的训练结果将始终保持一致
    cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题

    # pass

#
# 同一个模型，同一个参数，结果差几个点，今天和昨天 0.8424, at 65 epoch

def ResNet18(in_ch = 3):
    return ResNet(ResidualBlock, [2, 2, 2, 2], 10, in_ch=in_ch)

if __name__ == "__main__":
    # 固定随机数
    init()

    # Device configuration.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = get_logger("one", save_path=".", level='INFO', file_name="log_one_model.txt")
    print_fn = logger.info

    # Hyper-parameters
    num_classes = 10
    batch_size = 100    # 测试 batch size 的影响 ！ 100, 128, 64
    learning_rate = 0.001    # 0.001
    num_epochs = 100

    # mean, std = {}, {}
    # mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = transforms.Compose([transforms.Pad(4),
                                    transforms.RandomHorizontalFlip(),  # ? 水平翻转
                                    transforms.RandomCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    # transforms.Normalize(mean["cifar10"], std["cifar10"])   # 0.8576
                                    ]
                                   )

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize([0.5, 0.5, 0.5],
                                                                                      [0.5, 0.5, 0.5])])

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = CIFAR10(root="/home/user/PycharmProjects/data/cifar", train=True, download=False,
                            transform=transform)
    # 图像变换，导致测试精度变差
    # test_dataset = CIFAR10(root="/home/user/PycharmProjects/data/cifar", train=False, download=False,
    #                        transform=transform)
    test_dataset = CIFAR10(root="/home/user/PycharmProjects/data/cifar", train=False, download=False,
                           transform=transform_test)

    # 计算需要多少次迭代
    n_iters = len(test_dataset) // batch_size

    # Data Loader.
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    # *********************
    # Make model.
    # 能用的全用 89.58 %
    # 不用调度器 88.71 % ,  0.874, at 96 epoch
    # 不用调度器 和 不用 Normalize 88.55 %
    # 换成 adam -> SGD 73.07 %
    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes).to(device) # resnet18

    # **************  官方模型  *******************
    # model = torchvision.models.resnet50(num_classes = 10, pretrained=False).cuda()  # 0.8424, at 65 epoch
    # 原始调度器  # top acc: 0.8329, at 95 epoch
    # 不用调度器 82.92 %, at 99 round.
    # model = torchvision.models.resnet18(num_classes = 10, pretrained=False).cuda()


    # *************** ssl ***************
    # model = ResNet50(n_class= 10, is_remix=False, is_gray= False).cuda() # 0.71 不用其他的技巧
    # model = ResModel(in_ch=3).to(device)  # 0.72  #  两个 ResBlock

    # Loss ans optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = None
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_epochs * n_iters)

    # Train the model.
    total_step = len(train_loader)
    curr_lr = learning_rate

    train(model,
          train_loader,
          num_epochs=num_epochs,
          print_fn=print_fn,
          # scheduler=scheduler,
          curr_lr=learning_rate
          )




    # # 测试模型精度
    # state = torch.load('ResNet18.ckpt')
    # model.load_state_dict(state)
    # test(model, test_loader)


    # Save the model
    # ResNet18 3.2MB
    # torch.save(model.state_dict(), 'ResNet18.ckpt')

