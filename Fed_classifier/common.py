import yaml
import os
import random
import numpy as np
import torch
from torch.backends import cudnn
import argparse
import json
import logging
from torchvision import transforms
from dataset.BasicDataset import BasicDataset
from dataset.fileDataset import fileDataset
from models.ResNet import ResNet18
from models.client import Client
from utils import get_public_data, cifar_noniid, get_user_data, \
    fmnist_noniid, cifar_noniid_dirichlet, fmnist_noniid_dirichlet, get_data_random_idx, get_lackdata_idx, \
    cifar_noniid_byclass, fmnist_noniid_byclass


def dict2namespace( config ):
    namespace = argparse.Namespace()    # ？
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def str2bool(v):        # 吸收
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_models( config, logger ):

    n_clients= config.n_clients

    # 准备数据
    path = config.dataset_path
    if config.noniid:
        # non iid   Dirichlet分布
        if config.dataset == "Cifar10":
            if config.dirichlet:
                # dirichlet 分布
                train_data_dict_list, test_data_dict_list = cifar_noniid_dirichlet(n_clients,
                                                                                   alpha= config.alpha,
                                                                                   path= path)
            else:
                # train_data_dict_list, test_data_dict_list = cifar_noniid(n_clients, path=path)
                train_data_dict_list, test_data_dict_list = cifar_noniid_byclass(config.n_cls, path=path)
        elif config.dataset == "FashionMNIST":
            if config.dirichlet:
                train_data_dict_list, test_data_dict_list = fmnist_noniid_dirichlet(n_clients, alpha=config.alpha, path= path)
            else:
                # train_data_dict_list, test_data_dict_list = fmnist_noniid(num_users= n_clients, path= path)
                train_data_dict_list, test_data_dict_list = fmnist_noniid_byclass(config.n_cls, path=path)  # 规定 10 client
    else:
        # iid
        if config.dataset == "Cifar10":
            train_data_dict_list = get_user_data(n_clients, train=True, dataname="Cifar10")
            test_data_dict_list = get_user_data(n_clients, train=False, dataname="Cifar10")
        elif config.dataset == "FashionMNIST":
            train_data_dict_list = get_user_data(n_clients, train=True, dataname="FashionMNIST")
            test_data_dict_list = get_user_data(n_clients, train=True, dataname="FashionMNIST")

    if config.dataset == "Cifar10":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),  # ? 水平翻转
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
        )
    elif config.dataset == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),  # ? 水平翻转
            transforms.RandomCrop(28),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # error 变为 3 通道
            transforms.Normalize(0.5, 0.5)
        ])

    is_gray = False
    if config.dataset == "FashionMNIST":
        is_gray = True

    # # if config.use_gen :
    #     # path_data_file = "/home/user/PycharmProjects/ACGAN_cifar10-master/save"
    # path_data_file = config.gen_path
    # # cifar10 扩散模型生成样本，400 epoch，5 用户，10w， 联邦训练的得到
    # g_dataset = fileDataset(path_data_file,
    #                         transform=transform,
    #                         is_gray=is_gray,
    #                         )
    # # 随机划分
    # dict_users = get_data_random_idx(g_dataset, config.n_block)
    # # dict_users = get_data_random_idx(t_dataset, 5)
    # data_list = []
    # for i in range(n_clients):
    #     idxs = dict_users[i]
    #     dataset = fileDataset(path_data_file,
    #                         transform=transform,
    #                         is_gray=is_gray,
    #                         idxs= idxs,
    #                         path_filter= config.save_dir + "/issue_data"
    #                         )
    #     data_list.append(dataset)

    clients_data_dtb = []   # 用户数据分布，最终是个二维列表

    # 初始化用户
    client_list = []
    for i in range(n_clients):

        in_ch = 3
        if config.dataset != "Cifar10":     # 注意此处
            in_ch = 1
        C_model = ResNet18(in_ch= in_ch)
        data = train_data_dict_list[i]["sub_data"]
        targets = train_data_dict_list[i]["sub_targets"]
        # test_data = test_data_dict_list[i]["sub_data"]
        # test_targets = test_data_dict_list[i]["sub_targets"]

        ################# 获取每类样本的数量，并保存在文件中
        # 用numpy简化统计数据分布的操作
        count = [0 for _ in range(config.classes)]
        for c in targets:  # lb_targets 为 0 ～ 9 ， 有操作
            count[c] += 1
        # out = {"distribution": count }
        clients_data_dtb.append(count)  # 用户数据分布
        # # if config.use_gen:
        # #     g_count = [0 for _ in range(10)]
        # #     for c in data_list[i].labels:  #
        # #         g_count[c] += 1
        # #     out["generated_data_distribution"] = g_count
        #
        # output_file = f"{config.save_dir}/client_data_statistics_{i}.json"
        # # if not os.path.exists(output_file):
        # #     os.makedirs(output_file, exist_ok=True)
        # with open(output_file, 'w') as w:
        #     json.dump(out, w)
        #################

        dataset = BasicDataset(data, targets, transform=transform,onehot= False)
        # if config.use_gen:
        #     dataset = dataset + data_list[i]
        # dataloader = torch.utils.data.DataLoader(dataset, config.batch_size,
        #                                          shuffle=True,
        #                                          num_workers= config.num_works)
        # test_dataloader = torch.utils.data.DataLoader(test_dataset, 256,
        #                                          shuffle=False,
        #                                          num_workers= config.num_works)

        client = Client(config,
                        C_model=C_model,
                        client_idx=i,
                        # dataloader= dataloader,
                        dataset = dataset,  # 自身的原始数据
                        # t_dataloader= test_dataloader,
                        logger= logger
                        )

        # Optimizers
        # optimizer_C = torch.optim.SGD(C_model.parameters(),lr=0.001) 0.01
        optimizer_C = torch.optim.Adam(C_model.parameters(), lr = config.lr)  #更适合resnet18
        client.set_optimizer( optimizer_C )
        client_list.append( client )

    # ==================== 生成数据集的处理 =====================
    path_data_file = config.gen_path
    # cifar10 扩散模型生成样本，400 epoch，5 用户，10w， 联邦训练的得到
    g_dataset = fileDataset(path_data_file,
                            transform=transform,
                            is_gray=is_gray,
                            )
    # 随机划分
    # dict_users = get_data_random_idx(g_dataset, config.n_block)
    # 按照分布划分，得到数据的索引
    if config.add == "add_random":
        dict_users = get_data_random_idx(g_dataset, config.n_block)
    elif config.add == "add_lack":
        dict_users = get_lackdata_idx(g_dataset, clients_data_dtb,
                                      n_clients, div= config.div, logger= logger)
    else :
        print("未实现")

    # dict_users = get_lackdata_idx(g_dataset, clients_data_dtb, n_clients)

    data_list = []
    for i in range(n_clients):
        idxs = dict_users[i]    # 数据集中相应的数据的索引
        dataset = fileDataset(path_data_file,
                            transform=transform,
                            is_gray=is_gray,
                            idxs= idxs,
                            path_filter= config.save_dir + "/issue_data"
                            )
        data_list.append(dataset)

    # =================== 保存数据分布 =================
    logger.info("---数据统计---")
    sum_data = 0
    for i in range(n_clients):
        out = {"distribution": clients_data_dtb[i] }
        if config.use_gen:
            g_count = [0 for _ in range(10)]
            for c in data_list[i].labels:  #
                g_count[c] += 1
            out["generated_data_distribution"] = g_count

        output_file = f"{config.save_dir}/client_data_statistics_{i}.json"
        # if not os.path.exists(output_file):
        #     os.makedirs(output_file, exist_ok=True)
        logger.info("client {}, origin data {}, add data {}".format(i,
                                                    sum( clients_data_dtb[i]),sum(g_count)))
        sum_data += sum(g_count)
        with open(output_file, 'w') as w:
            json.dump(out, w)
    logger.info("---sum add data: {} ----".format( sum_data ))

    return client_list , transform, data_list


def init():

    # 添加参数
    parser = argparse.ArgumentParser(description=globals()["__doc__"])  # ？
    parser.add_argument("--name", type=str, default="Fed-Dif", help="FedAvg/FedProx/FedNova/Scaffold")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="save", help="Path to save results")
    parser.add_argument("--n_clients", type=int, default=10, help="number of clients，5，10，20")
    parser.add_argument("--noniid", type=str2bool, default=True, help="noniid or not")  # 把 True 固定了 ，type的问题，解决，添加str2bool函数
    parser.add_argument("--dirichlet", type=str2bool, default=False, help="dirichlet or not") # 有问题
    # parser.add_argument("--dirichlet",action="store_true", help="dirichlet or not")
    parser.add_argument("--alpha", type=float, default=1.0, help="control dirichlet")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path of dataset")
    parser.add_argument("--dataset", type=str, default="Cifar10", help="Cifar10/FashionMNIST")
    parser.add_argument("--num_works", type=int, default=8, help="Number of classes")
    parser.add_argument("--n_cls", type=int, default=4, help="n classes per client have: 2 | 4")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size: 128")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--round", type=int, default=100, help="communication round")
    parser.add_argument("--local_epoch", type=int, default=5, help="local epoch")

    # 生成数据集
    parser.add_argument("--use_gen", type=str2bool, default=True, help="user generated dataset or not")
    parser.add_argument("--gen_path", type=str, default="/home/user/Gen_Data/cifar10/cifar10-10c-400r-1e-alpha-0.1-10w",
                        help="Path of generated dataset")
    # parser.add_argument("--gen_path", type=str, default="/home/user/PycharmProjects/ACGAN_cifar10-master/save_img",
    #                     help="Path of generated dataset")


    parser.add_argument("--add", type=str, default="add_random", help="数据扩充类型：add_random | add_lack")
    parser.add_argument("--div", type=int, default=10, help="add_lack 中的平均方式：10，2，4")   # 用户自身的样本总数除以多少类
    parser.add_argument("--n_block", type=int, default=10, help="add_random分的块数，和用户数量相关")
    # 过滤
    parser.add_argument("--filter_round", type=int, default=99999, help="filter")   # 在第几轮开始使用过滤
    parser.add_argument("--save_n_sample", type=str2bool, default=True, help="save noise sample")   # 是否保存过滤的样本

    # args = parser.parse_args()

    # # 读取配置文件
    # with open(os.path.join("configs", args.config), "r") as f:
    #     config = yaml.safe_load(f)
    # new_config = dict2namespace(config)     #

    new_config = parser.parse_args()

    # 保存路径
    save_dir = None
    if new_config.noniid == False:
        save_dir = os.path.join(new_config.save_dir,
                                new_config.name + f"_{new_config.dataset}" + \
                                f"_{new_config.n_clients}" + f"_iid" )
    elif new_config.noniid and new_config.dirichlet == False:
        save_dir = os.path.join(new_config.save_dir,
                                new_config.name + f"_{new_config.dataset}" + \
                                f"_{new_config.n_clients}" + f"_noniid" )
    elif new_config.noniid and new_config.dirichlet:
        save_dir = os.path.join(new_config.save_dir,
                                new_config.name  + f"_{new_config.dataset}" + \
                                f"_{new_config.n_clients}" + \
                                f"_dirichlet_alpha_{new_config.alpha}")
    new_config.save_dir = save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)    # True：创建目录的时候，如果已存在不报错。

    # 获取设备信息
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # 将配置信息保存
    with open(os.path.join(new_config.save_dir, "config.yml"), "w") as f:
        yaml.dump(new_config, f, default_flow_style=False)

    seed = new_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True  # 随机数种子seed确定时，模型的训练结果将始终保持一致
    cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    return new_config
