import argparse
import json
import os
import random

import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image      # 不了解
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from core_fedavg import fedavg_gan, fedavg_classifier
from models.client import Client
from utils import get_user_data, fmnist_noniid, get_logger, cifar_noniid
from dataset.BasicDataset import BasicDataset
from dataset.fileDataset import fileDataset
from models.simple_model import Discriminator, Generator, MLP , ResModel
from torch.backends import cudnn    # !!! 作用


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def load_model(logger):
    n_client= 5 #10  # 5  20
    # 准备数据
    # iid
    # train_data_dict_list = get_user_data(5, train=True, dataname="FashionMNIST")
    # val_data_dict_list = get_user_data(5, train=False, dataname="FashionMNIST")
    path = "/home/user/PycharmProjects/data"
    train_data_dict_list, _ = cifar_noniid(n_client, path= path)

    # non iid
    # train_data_dict_list, _ = fmnist_noniid(num_users= n_client)  # 5 10 20

    # 生成样本
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_noniid_imgs", transform=transform)
    # 共 2**10 * 10 * 10 样本  扩散模型  50 epoch 5 user
    t_dataset = fileDataset("/home/user/PycharmProjects/data/gen_data_diffusion/fmnist/generate_noniid_imgs_5user_49epoch",
                            transform=transform)
    # 共 2**10 * 10 * 10 样本  扩散模型  150 epoch 5 user
    # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_noniid_imgs_10w_5user_150epoch", transform=transform)
    # data_list = torch.utils.data.random_split(t_dataset, [2**10 * 10 for _ in range(n_client)])
    data_list = torch.utils.data.random_split(t_dataset, [2 ** 10 * 10 for _ in range(10)])
    # 5 user 使用全部数据 2 ** 10 * 10 * 10
    # data_list = torch.utils.data.random_split(t_dataset, [2 ** 10 * 10 * 2 for _ in range(5)])


    # Confusion model生成的样本
    # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_imgs", transform=transform)
    # dataloader_list = [torch.utils.data.DataLoader(t_dataset, 64, shuffle=True) for _ in range(5)]

    # 初始化用户
    client_list = []

    for i in range(n_client):
        G = Generator()
        D = Discriminator()
        # C_model = MLP()
        C_model = ResModel(in_ch= 3)
        G.weight_init(mean=0.0, std=0.02)
        D.weight_init(mean=0.0, std=0.02)
        data = train_data_dict_list[i]["sub_data"]
        targets = train_data_dict_list[i]["sub_targets"]
        # 获取每类样本的数量，并保存在文件中
        count = [0 for _ in range(10)]
        for c in targets:  # lb_targets 为 0 ～ 9 ， 有操作
            count[c] += 1
        out = {"distribution": count}
        output_file = "save/client_data_statistics_%d.json"%(i)
        # if not os.path.exists(output_file):
        #     os.makedirs(output_file, exist_ok=True)
        with open(output_file, 'w') as w:
            json.dump(out, w)


        transform = transforms.Compose( [transforms.Resize(32), transforms.ToTensor()] )
        dataset = BasicDataset(data, targets, transform=transform,onehot= False)
        # t_dataset = fileDataset("data/generate_imgs",transform=transform)

        # 只使用自身数据
        # dataloader = torch.utils.data.DataLoader(dataset, opt.batch_size,
        #                                          shuffle=True, )
        # 混合生成样本和真实样本
        dataloader = torch.utils.data.DataLoader(dataset + data_list[i], opt.batch_size,
                                                 shuffle=True)
        client = Client(G, D,C_model=C_model, client_idx=i, dataloader= dataloader, logger= logger)

        # Optimizers
        optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))  # 优化器的原理和特点
        optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_C = torch.optim.SGD(C_model.parameters(),lr=0.03)
        client.set_optimizer(optimizer_G, optimizer_D, optimizer_C)
        client_list.append(client)

    return client_list

def init():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True  # 随机数种子seed确定时，模型的训练结果将始终保持一致
    cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题

    # pass


def main():
    init()

    IMG_PATH = "images"
    os.makedirs(IMG_PATH, exist_ok=True)
    os.makedirs("save-local-epoch-2", exist_ok=True)
    os.makedirs("data/generate_imgs", exist_ok=True)

    logger = get_logger("fedgan", save_path=".", level='INFO', file_name="log_fedDefusion.txt")
    model_list = load_model(logger)
    # fedavg_gan(model_list, logger= logger)
    fedavg_classifier(model_list, logger=logger)

if __name__ == "__main__":
    main()
