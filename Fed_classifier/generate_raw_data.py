import os

import torchvision.utils as tvu
from torchvision.datasets import CIFAR10, FashionMNIST
import torch
from torchvision import transforms

from dataset.fileDataset import fileDataset

if __name__ == "__main__":
    # path = "./data"
    # save_name = "cifar10"

    path = "./data"
    save_name = "fmnist"

    # path_data_file ="/home/user/GAN_Data/Cifar10/single_400r_10w"
    # save_name = "cifar10-acgan"
    os.makedirs(save_name, exist_ok= True)

    # transform = transforms.Compose([
    #     transforms.Resize(32),
    #     transforms.ToTensor(),
    # ])



    # dataset = CIFAR10(root=path,  download=False, train=True,transform=transforms.ToTensor() )
    dataset = FashionMNIST(root=path,  download=False, train=True, transform=transforms.ToTensor() )

    # transform = transforms.Compose([
    #     transforms.Resize(28),
    #     transforms.ToTensor(),
    # ])
    # dataset = fileDataset(path_data_file,
    #                         transform=transform,
    #                         is_gray=False,
    #                         )

    dataloader = torch.utils.data.DataLoader(dataset,
                                           # batch_size=4096,
                                           batch_size=64 ,
                                           num_workers=10,
                                            shuffle=True)
    temp = 0
    for x, y in dataloader:
        for j in range(x.size(0)):
            tvu.save_image(
                x[j], os.path.join(save_name, f"{temp}.png")
            )
            temp += 1