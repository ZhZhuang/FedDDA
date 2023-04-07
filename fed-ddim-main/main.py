import argparse
import json
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
from torchvision.transforms import transforms

from datasets.BasicDataset import BasicDataset
from runners.diffusion import Diffusion
from client import Client
from utils import cifar_noniid, cifar_noniid_dirichlet, fmnist_noniid, fmnist_noniid_dirichlet, cifar_noniid_byclass, \
    fmnist_noniid_byclass

torch.set_printoptions(sci_mode=False)  # ？


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    # parser.add_argument(
    #     "--config", type=str, required=True,default="configs/cifar10.yml", help="Path to the config file"
    # )
    parser.add_argument(
        "--config", type=str, default="cifar10.yml", help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default="save_fed",
        help="A string for documentation purpose. "
             "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--generate", action="store_true")  #  new
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    ###
    parser.add_argument(
        "--resume_training", action="store_true",
        # default= True,
        help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        default= True,
        help="No interaction. Suitable for Slurm Job launcher", # 考虑周到
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",  # ？
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",  # ？
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",    # ？
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)     #

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    # overwrite = True
                    overwrite = False
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                # else:
                #     print("Folder exists. Program halted.")
                #     sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        # overwrite = True
                        overwrite = False
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    # else:
                    #     print("Output image folder exists. Program halted.")
                    #     sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # # set random seed   固定随机数，样本生成时也会固定
    if not args.sample:     # 训练时固执随机
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # # set random seed   固定随机数，样本生成时也会固定
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace




def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    # import torch, gc
    # gc.collect()
    # torch.cuda.empty_cache()

    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample()
        elif args.test:
            runner.test()
        else:   # 联邦训练
            n_clients = config.fed.n_clients
            clients = []
            # path = "/home/user/PycharmProjects/data/cifar"
            # path = os.path.join(args.exp, "datasets", "cifar10"),

            if config.data.dataset == "CIFAR10":
                if config.fed.dirichlet:
                    # dirichlet 分布
                    logging.info("Clients:{}，Dirichlet 分布，alpha: {}".format(n_clients, config.fed.alpha))
                    train_data_dict_list, _ = cifar_noniid_dirichlet(n_clients, alpha=config.fed.alpha )
                else:
                    # train_data_dict_list, _ = cifar_noniid(n_clients)
                    train_data_dict_list, _ = cifar_noniid_byclass(num_class= config.fed.num_class)
            elif config.data.dataset == "FashionMNIST":
                if config.fed.dirichlet:
                    logging.info("Clients:{}，Dirichlet 分布，alpha: {}".format(n_clients, config.fed.alpha))
                    train_data_dict_list, _ = fmnist_noniid_dirichlet(n_clients, alpha=config.fed.alpha )
                else:
                    # train_data_dict_list, _ = fmnist_noniid(num_users=n_clients)
                    train_data_dict_list, _ = fmnist_noniid_byclass(num_class= config.fed.num_class)

            # tf
            if config.data.random_flip is False:
                tran_transform = transforms.Compose(
                    [
                     transforms.Resize(config.data.image_size),
                     transforms.ToTensor()]
                )
            else:
                tran_transform = transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])    # 尝试有没有效果，加上探究是否是其导致不收敛
                        # 加上 Normalize 反而会影响训练效果，加大loss，在联邦学习的情况下
                    ]
                )
                # test_transform = transforms.Compose(
                #     [transforms.Resize(config.data.image_size), transforms.ToTensor()]
                # )

            for i in range(n_clients):
                data = train_data_dict_list[i]["sub_data"]
                targets = train_data_dict_list[i]["sub_targets"]

                ################# 获取每类样本的数量，并保存在文件中
                count = [0 for _ in range(10)]
                for c in targets:  # lb_targets 为 0 ～ 9 ， 有操作
                    count[c] += 1
                out = {"distribution": count}
                out_path = "save_distribution"
                if not os.path.exists(out_path):
                    os.makedirs(out_path, exist_ok=True)
                output_file = f"{out_path}/client_data_statistics_{i}.json"     # args.log_path,
                with open(output_file, 'w') as w:
                    json.dump(out, w)
                #################
                dataset = BasicDataset(data, targets, transform=tran_transform, onehot=False)

                client = Client(config, dataset)
                clients.append(client)
            runner.fed_train(clients)   # 运行
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
