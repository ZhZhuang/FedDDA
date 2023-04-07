import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import BatchSampler
# 吸收
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.utils import save_image

from dataset.BasicDataset import BasicDataset
from dataset.fileDataset import fileDataset
from utils import get_user_data

cuda = True
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

# 装饰器语法， 此处的装饰器和上下文管理器很像、 enter，exit
@torch.no_grad()
def sample_image(n_row, batches_done, generator= None, generator2 = None, path= "images"):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    generator = generator
    # Sample noise
    # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    z_ = torch.randn(( n_row ** 2,100 )).view(-1, 100, 1, 1)
    # z = Variable(FloatTensor(z_))
    z = z_.cuda()
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = onehot[labels] # 生成独热 10 * 1 * 1
    # labels = Variable(LongTensor(labels))
    labels = labels.type(FloatTensor).cuda()
    gen_imgs = generator(z, labels)
    # torchvision 中自带了图片保存函数，对于pytorch， torchvision的功能不了解
    save_image(gen_imgs.data, "%s/%d.png" % (path, batches_done), nrow=n_row, normalize=True)
    if not generator2 is None:
        gen_imgs = generator2(z, labels)
        save_image(gen_imgs.data, "%s/%dtest.png" % (path, batches_done), nrow=n_row, normalize=True)

# make_grid() 的作用
@torch.no_grad()
def gen_image(n_imgs, n_class, generator= None,  path= "images"):
    generator = generator
    # Sample noise
    # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    for c in range(n_class):
        z_ = torch.randn(( n_imgs,100 )).view(-1, 100, 1, 1)
        z_ = z_.cuda()
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([c for _ in range(n_imgs)])  # n * n
        labels_ = onehot[labels] # 生成独热 10 * 1 * 1
        # labels = Variable(LongTensor(labels))
        labels_ = labels_.type(FloatTensor).cuda()
        gen_imgs = generator(z_, labels_)
        for num in range(len(gen_imgs)):
            save_image(gen_imgs.data[num], "%s/%d_%d.png" % (path, c, num),padding= 0,normalize=True)
            # save_image(gen_imgs.data[num], "%s/%d_%d.png" % (path, labels[num], num), nrow=n_row, normalize=True)


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            #print('done')
            w_avg[k] += w[i][k]     # 有没有不能汇聚的层 ？
        # w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = torch.true_divide(w_avg[k], len(w))  #兼容pytorch 1.6
    return w_avg

# 每轮得到的GAN的质量相差很大，不是时间越久越好，质量好与坏出现一种交叉出现的现象
# 局部更新
def fedavg_gan( clients, logger = None ):
    print_fn = print if logger is None else logger.info
    print_fn("================== fedavg train ========================")
    # print_fn(KT_pFL_params)
    round = 200 # 50
    N_private_training_round = 1    # 默认是 1,
    # w_global = clients[0].model.state_dict()
    w_locals = [clients[0].generator.state_dict() for i in range(len(clients))] #也可以b = [0]*10
    # w_optims = [clients[0].optimizer_G.state_dict() for i in range(len(clients))]
    for r in range(round):
        print_fn("round : {0}".format(r))
        # 本地训练
        for index, client in enumerate(clients):
            print("model {0} starting loacal train... ".format(index))
            for i in range(N_private_training_round):  # 1 、 10\ 5
                client.train_gan()  # 对自身私有数据集训练一遍
            w_locals[index] = copy.deepcopy(client.generator.state_dict())
            # w_optims[index] = copy.deepcopy(client.optimizer_G.state_dict())
            print("model {0} done local train. \n".format(index))

        # 联邦聚合
        with torch.no_grad():
            w_global = FedAvg(w_locals)
            # w_optim_global = FedAvg(w_optims)
        for index in range(len(clients)):
            # clients[index].model.load_state_dict(w_global)
            clients[index].generator.load_state_dict(w_global,strict = False)  # 适合部分加载模型
            # clients[index].optimizer_G.load_state_dict(w_optim_global,strict = False)
            # clients[index].model.state_dict().update(w_global)  # 有没有区别
            # model_dict = clients[index].model.state_dict()
            # model_dict.update(w_global)
            # clients[index].model.load_state_dict(model_dict)
        # # 联邦聚合
        # with torch.no_grad():
        #     w_global = FedAvg(w_locals)
        # # for index, client in enumerate(clients):
        # #     client.model.load_state_dict(w_global)    #是引用
        # for index in range(len(clients)):
        #     clients[index].model.load_state_dict(w_global)
        # if r % 10 == 0:
        #     torch.save(clients[0].generator.state_dict(), "save/generate_%d.pth" % (r))

        torch.save(clients[0].generator.state_dict(), "save/generate_%d.pth" % (r))
        sample_image(n_row=10, batches_done=r, generator=clients[0].generator)

    # 随机生成样本
    gen_image(n_imgs=1000, n_class=10, generator=clients[0].generator, path="data/generate_imgs")
    print_fn("================== fedavg gan end ========================")


# # 固定随机数，否者差距较大
# def fedavg_classifier( clients, logger = None ):
#     print_fn = print if logger is None else logger.info
#     print_fn("================== fedavg train ========================")
#     # round = 100  #20
#     round = 100
#     num_client = len(clients)
#     acc_list = [0 for i in range(len(clients))]
#     N_private_training_round = 1    # 默认是 1,
#     w_locals = [ clients[0].c_model.state_dict() for i in range(len(clients)) ] #也可以b = [0]*10
#     transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
#
#     # 加一步数据分割
#     # t_dataset = fileDataset("data/generate_imgs", transform=transform)
#     # data_list = torch.utils.data.random_split(t_dataset, [2400 for _ in range(5)])
#     # dataloader_list = [torch.utils.data.DataLoader(data_list[i], 64, shuffle=True) for i in range(5) ]
#     # t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=64, num_workers=10, shuffle=True)
#
#     # gan sample
#     # t_dataset = fileDataset("data/generate_imgs", transform=transform)
#     # dataloader_list = [torch.utils.data.DataLoader(t_dataset, 64, shuffle=True) for _ in range(5)]
#     # t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=64, num_workers=10, shuffle=True)
#
#
#     # confusion sample
#     t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_imgs", transform=transform)
#     # data_list = torch.utils.data.random_split(t_dataset, [2400 for _ in range(5)])
#     # dataloader_list = [torch.utils.data.DataLoader(t_dataset, 64, shuffle=True) for _ in range(5)]
#     t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=64, num_workers=10, shuffle=True)
#
#     # DataLoader 的加载方式 Sampler
#     # t_dataset = fileDataset("data/generate_imgs", transform=transform)
#     # data_sampler = RandomSampler(t_dataset, replacement=True, num_samples=12000)  # 生成索引
#     # batch_sampler = BatchSampler(data_sampler, batch_size=64, drop_last=True)  # ?
#     # t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_sampler=batch_sampler, num_workers= 8)
#
#     ###### 验证集
#     # val_data_dict_list = get_user_data(5, train=False, dataname="FashionMNIST")
#     # data = val_data_dict_list[0]["sub_data"]
#     # targets = val_data_dict_list[0]["sub_targets"]
#     # v_dataset = BasicDataset(data, targets, transform=transform, onehot=False)
#     v_dataset = FashionMNIST("./data", train=False, download=False, transform=transform)
#     v_dataloader = torch.utils.data.DataLoader(v_dataset, 1028, shuffle=True)
#     top_acc = 0.
#     top_r = 0
#     for r in range(round):
#         print_fn("round : {0}".format(r))
#
#         # 本地训练
#         for index, client in enumerate(clients):
#             print("model {0} starting loacal train... ".format(index))
#             for i in range(N_private_training_round):  # 1 、 10\ 5
#                 # 顺序训练fmnist和生成的数据集，比放一起训练结果要好 ！
#                 client.train_classifier_1()  # 对自身私有数据集训练一遍
#                 # client.train_classifier_1(dataloader_list[i])  #
#                 # client.train_classifier(t_dataloader)  # 对自身私有数据集训练一遍
#             w_locals[index] = copy.deepcopy( client.c_model.state_dict() )
#             print("model {0} done local train. \n".format(index))
#
#
#         # # 联邦聚合
#         with torch.no_grad():
#             w_global = FedAvg(w_locals)
#         #     # w_optim_global = FedAvg(w_optims)
#         for index in range(len(clients)):
#             # clients[index].model.load_state_dict(w_global)
#             clients[index].c_model.load_state_dict(w_global,strict = False)  # 适合部分加载模型
#
#         # 过滤劣质样本
#         temp = clients[0].filter_data(t_dataloader)
#
#         # 测试
#         print_fn("round {} evaluate... ".format(r))
#         for index, client in enumerate(clients):
#             # print("model {0} evaluate... ".format(index))
#             acc_list[index] = client.evaluate(v_dataloader)  # 对自身私有数据集训练一遍
#         print_fn("round {} mean acc {}".format(r,sum( acc_list )/num_client))
#         if sum( acc_list )/num_client > top_acc:
#             top_acc = sum( acc_list )/num_client
#             top_r = r
#         # clients[0].evaluate(v_dataloader)
#
#     print_fn(f"top acc : {top_acc}, at {top_r} round!")
#     print_fn("================== fedavg classifier end ========================")
#



# 固定随机数，否者差距较大
def fedavg_classifier( clients, logger = None,  round = 100 ):
    print_fn = print if logger is None else logger.info
    print_fn("================== fedavg train ========================")
    # round = 100  #20， 200

    num_client = len(clients)
    acc_list = []
    N_private_training_round = 1    # 默认是 1,
    w_locals = [ clients[0].c_model.state_dict() for i in range(len(clients)) ] #也可以b = [0]*10
    # transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    transform = transforms.Compose([transforms.Pad(4),
                                    transforms.RandomHorizontalFlip(),  # ? 水平翻转
                                    transforms.RandomCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ]
                                   )

    # confusion sample
    # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_noniid_imgs", transform=transform)
    # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_noniid_imgs_10w_5user_150epoch", transform=transform)
    # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_iid_imgs", transform=transform)
    # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_imgs", transform=transform)
    # data_list = torch.utils.data.random_split(t_dataset, [2400 for _ in range(5)])
    # dataloader_list = [torch.utils.data.DataLoader(t_dataset, 64, shuffle=True) for _ in range(5)]
    # t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=64, num_workers=10, shuffle=True)

    ###### 验证集
    # val_data_dict_list = get_user_data(5, train=False, dataname="FashionMNIST")
    # data = val_data_dict_list[0]["sub_data"]
    # targets = val_data_dict_list[0]["sub_targets"]
    # v_dataset = BasicDataset(data, targets, transform=transform, onehot=False)
    v_dataset = CIFAR10(root="/home/user/PycharmProjects/data/cifar",
                        train=False, download=False,
                        transform=transform)
    # v_dataset = FashionMNIST("./data", train=False, download=False, transform=transform)
    v_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size= 1028, shuffle=True)
    top_acc = 0.
    top_r = 0
    remove_list = []
    for r in range(round):
        start_time = time.time()
        print_fn("round : {0}".format(r))
        # 卸磨杀驴
        # if len(remove_list) > 0:
        #     print("del num ： {} ！".format(len(remove_list)))
        #     t_dataset.remove_sample(remove_list)
        #     t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=64,
        #                                                num_workers=10, shuffle=True)
        # 本地训练
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)    # 选择部分用户
        for index, client in enumerate(clients):
            print("model {0} starting loacal train... ".format(index))
            for i in range(N_private_training_round):  # 1 、 10\ 5
                # 顺序训练fmnist和生成的数据集，比放一起训练结果要好 ！
                client.train_classifier_1()  # 对自身私有数据集训练一遍
                # client.train_classifier_2(t_dataloader)

                # client.train_classifier_1(dataloader_list[i])  #
                # client.train_classifier(t_dataloader)  # 效果最差的方式， 同事训练自身的数据和 生成数据
            w_locals[index] = copy.deepcopy( client.c_model.state_dict() )
            print("model {0} done local train. \n".format(index))


        # # 联邦聚合
        with torch.no_grad():
            w_global = FedAvg(w_locals)
        #     # w_optim_global = FedAvg(w_optims)
        for index in range(len(clients)):
            clients[index].c_model.load_state_dict(w_global,strict = True)  # 适合部分加载模型

        # # 过滤劣质样本
        # if r > 2 :     #  30， 10  20 判断每轮应该删除的数量
        #     n_gen_samplers =  len(t_dataset)
        #     remove_list =  clients[0].filter_data_CPL(t_dataloader, n_gen_samplers)
        #     # remove_list = clients[0].filter_data(t_dataloader)

        # 测试
        print_fn("round {} evaluate... ".format(r))
        # for index, client in enumerate(clients):
        #     # print("model {0} evaluate... ".format(index))
        #     acc_list[index] = client.evaluate(v_dataloader)  # 对自身私有数据集训练一遍
        acc_temp =clients[0].evaluate(v_dataloader)
        acc_list.append( acc_temp )
        dur_time = time.time() - start_time
        print_fn("round {} mean acc {}, time: {} sec a round".format(r, acc_temp, dur_time))
        if acc_temp > top_acc:
            top_acc = acc_temp
            top_r = r
        # clients[0].evaluate(v_dataloader)

    print_fn("测试结果：")
    print_fn(acc_list)
    print_fn(f"top acc : {top_acc}, at {top_r} round!")
    print_fn("================== fedavg classifier end ========================")

