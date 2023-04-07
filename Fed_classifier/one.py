import time

import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST

from models.ResNet import ResNet18, train
from models.client import Client
from models.resnet50 import build_ResNet50, ResNet50
from models.simple_model import ResModel, Generator, Discriminator
import torch.nn.functional as F
import argparse

from models.wrn import build_WideResNet
from utils import get_logger

def test(model , test_loader):
    # model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return correct / total


# For updating learning rate.
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 训练模型
def train(model, train_loader,
          num_epochs= 80,
          print_fn= None,
          scheduler= None,
          curr_lr= 0.001,
          test_loader= None):
    if print_fn == None:
        print_fn = print

    top_acc = 0.0
    top_epoch = 0
    acc_list = []

    criterion = torch.nn.CrossEntropyLoss()

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
                print_fn('Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                            loss.item()))

            # if scheduler != None:
            #     scheduler.step()
            #     # print(optimizer.param_groups[0]["lr"])

        # Decay learning rate.

        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)


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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()


    logger = get_logger("one", save_path=".", level='INFO', file_name="log_one_model.txt")
    print_fn = logger.info
    # transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    #         )
    # transform = transforms.Compose([transforms.Resize(32),
    #                                 transforms.ToTensor(),]
    #                                )

    # transform = transforms.Compose([transforms.Resize(32),
    #                                 transforms.ToTensor(),
    #                                 # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                                 ]
    #                                )
    # transform = transforms.Compose([transforms.Pad(4),
    #                                transforms.RandomHorizontalFlip(),   # ?
    #                                transforms.RandomCrop(32),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    transform = transforms.Compose([transforms.Pad(4),
                                   transforms.RandomHorizontalFlip(),   # ?
                                   transforms.RandomCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])])


    train_dataset = FashionMNIST("./data", train=True, download=False, transform=transform)
    test_dataset = FashionMNIST("./data", train=False, download=False, transform=transform)  # 1w 张
    # dataset = CIFAR10(root="/home/user/PycharmProjects/data/cifar", train=True, download=False, transform=transform)
    # test_dataset = CIFAR10(root="/home/user/PycharmProjects/data/cifar", train=False, download=False, transform=transform)


    # C_model = ResModel(in_ch= 3) # 0.72
    # top acc: 0.8251, at 88 round 数据加了更多的操作
    # C_model = ResNet50(n_class= 10, is_remix=False, is_gray= False) # 0.71 不用其他的技巧
    C_model = ResNet18(in_ch=1).cuda()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 64,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 64,
                                             shuffle=True)

    # client = Client(opt,C_model=C_model)
    optimizer = torch.optim.Adam(C_model.parameters(), lr=0.001)
    # optimizer_C = torch.optim.SGD(C_model.parameters(), lr=0.001, momentum= 0.9, weight_decay= 0.0005)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(C_model,
          train_dataloader,
          num_epochs=100,
          print_fn=print_fn,
          # scheduler=scheduler,
          curr_lr=0.001,
          test_loader= test_loader
          )
