import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST,CIFAR10


# def getStat(train_data):
#     '''
#     Compute mean and variance for training data
#     :param train_data: 自定义类Dataset(或ImageFolder即可)
#     :return: (mean, std)
#     '''
#     print('Compute mean and variance for training data.')
#     print(len(train_data))
#     train_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=1, shuffle=False, num_workers=0,
#         pin_memory=True)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     for X, _ in train_loader:
#         for d in range(3):
#             mean[d] += X[:, d, :, :].mean()
#             std[d] += X[:, d, :, :].std()
#     mean.div_(len(train_data))
#     std.div_(len(train_data))
#     return list(mean.numpy()), list(std.numpy())

def getStat(train_data, ch = 3):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    # dataset = train_data.data
    train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=1, shuffle=False, num_workers=0,
                     pin_memory=True)
    mean = torch.zeros(ch)   #  FashionMNIST 只有一个通道
    std = torch.zeros(ch)
    for X, _ in train_loader:
        for d in range(ch):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    # train_dataset = ImageFolder(root=r'D:\cifar10_images\test', transform=None)
    # train_dataset = FashionMNIST("/home/user/PycharmProjects/data/cifar", train=False, download=False,transform=transforms.ToTensor() )
    train_dataset = CIFAR10("/home/user/PycharmProjects/data/cifar",
                                 train=True,
                                 download=False,
                                 transform=transforms.ToTensor()
                            )
    # train_dataset = FashionMNIST("./data", train=True, download=True )
    test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    print(getStat(train_dataset))
    # for X, y in test_dataloader:#  FashionMNIST 的计算结果
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")# Compute mean and variance
    #     print(f"Shape of y: {y.shape} {y.dtype}")# for training data.
    #     break


# Fashion MNIST  ([0.28604063], [0.32045463])