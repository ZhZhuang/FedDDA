import copy
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import BatchSampler
# 吸收
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10


# def FedAvg(w):
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             #print('done')
#             w_avg[k] += w[i][k]     # 有没有不能汇聚的层 ？
#         # w_avg[k] = torch.div(w_avg[k], len(w))
#         w_avg[k] = torch.true_divide(w_avg[k], len(w))  #兼容pytorch 1.6
#     return w_avg


# 按照权重聚合
def FedAvg(w, wl= None):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0 :
                w_avg[k] = w_avg[k] * wl[i]
            #print('done')
            else:
                w_avg[k] += w[i][k] * wl[i]     # 有没有不能汇聚的层 ？
        # w_avg[k] = torch.div(w_avg[k], len(w))
        # w_avg[k] = torch.true_divide(w_avg[k], len(w))  #兼容pytorch 1.6
    return w_avg

# 固定随机数，否者差距较大
def fed_train(
            config,
            clients,
            logger = None,
            transform = None,
            data_list = None):

    print_fn = print if logger is None else logger.info
    print_fn(f"================== {config.name} train ========================")
    # round =

    round = config.round    # 100  #20， 200
    local_epoch = config.local_epoch
    num_client = len(clients)
    acc_list = []
    w_locals = [ clients[0].c_model.state_dict() for i in range(len(clients)) ] #也可以b = [0]*10


    ###### 验证集
    if config.dataset == "Cifar10":
        v_tf = transforms.Compose([transforms.Resize(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                          ])
        v_dataset = CIFAR10(root=config.dataset_path,
                            train=False,
                            download=False,
                            transform=v_tf)
    elif config.dataset == "FashionMNIST":
        v_tf = transforms.Compose([transforms.Resize(28),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5])
                                          ])
        v_dataset = FashionMNIST(root=config.dataset_path,
                                 train=False,
                                 download=False,
                                 transform=v_tf)
    # v_dataset = FashionMNIST("./data", train=False, download=False, transform=transform)
    v_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size= 512,
                                               num_workers=config.num_works,
                                                shuffle=False)
    top_acc = 0.
    top_r = 0
    remove_list = [ [] for _ in range(num_client) ]

    # # # # 加载预训练模型
    # state = torch.load(os.path.join(config.save_dir, "ckp_99_round.pth"))
    # for index in range(len(clients)):
    #     clients[index].c_model.load_state_dict(state)  # 适合部分加载模型
    #     print_fn("client {} 加载模型完毕！".format(index))

    for r in range(round):
        start_time = time.time()
        print_fn("round : {0}".format(r))

        # 本地训练
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)    # 选择部分用户
        for index, client in enumerate(clients):
            print("model {0} starting loacal train... ".format(index))
            t_dataset = data_list[index] + client.dataset
            client.num_sample = len(t_dataset)
            t_dataloader = torch.utils.data.DataLoader(t_dataset,
                                                       batch_size=config.batch_size,
                                                       num_workers=config.num_works, shuffle=True)
            for i in range(local_epoch):  # 1 、 10\ 5
                # client.train_classifier_1()  # 对自身私有数据集训练一遍
                client.train_classifier_2(t_dataloader)  #!!
                # client.train_withfocalloss(t_dataloader)
                # client.train_classifier_1(dataloader_list[i])  #
                # client.train_classifier(t_dataloader)  # 效果最差的方式， 同事训练自身的数据和 生成数据
            w_locals[index] = copy.deepcopy( client.c_model.state_dict() )
            print("model {0} done local train. \n".format(index))

        # 总数据量
        total_data_points = sum([clients[i].num_sample for i in range(num_client)])
        # 用户数据量的比例
        fed_avg_freqs = [clients[i].num_sample / total_data_points for i in range(num_client)]

        # # 联邦聚合
        with torch.no_grad():
            w_global = FedAvg(w_locals, fed_avg_freqs)
        #     # w_optim_global = FedAvg(w_optims)
        for index in range(len(clients)):
            clients[index].c_model.load_state_dict(w_global,strict = True)  # 适合部分加载模型

        # # 保存模型参数
        # if r == 99:
        #     torch.save(w_global, os.path.join(config.save_dir, f"ckp_{r}_round.pth"))
        # 保存模型参数
        if (r + 1) % 10 == 0 and r > 100 :
            state= [w_global, r]
            torch.save(state, os.path.join(config.save_dir, f"ckp.pth"))

        # 过滤劣质样本
        if r >= config.filter_round :     #  30， 10  20 判断每轮应该删除的数量
            for index in range(num_client):
                # 生成数据的 dataloader
                g_dataloader = torch.utils.data.DataLoader(data_list[index],
                                                           # batch_size=4096,
                                                           batch_size=4096 ,
                                                           num_workers=config.num_works, shuffle=True)
                remove_list[index] = clients[index].filter_with_cleanlab(g_dataloader)
                # remove_list = clients[0].filter_data(t_dataloader)

                if len(remove_list[index]) > 0:
                    print("client: {},del num: {} ！".format(index, len(remove_list[index])))
                    # data_list[index].remove_sample(remove_list)   一个小失误改了一晚上！
                    data_list[index].remove_sample(remove_list[index],
                                                   is_save=config.save_n_sample
                                                   )
                    # del remove_list[index]    # 导致数组少元素，索引越界

                    # # 生成样本的权重， focal loss 要用
                    # count = [0 for _ in range(config.classes)]
                    # for c in data_list[index].labels:  #
                    #     count[c] += 1
                    # t_s = sum(count)
                    # w = [i / t_s for i in count]
                    # clients[index].w_gen = w

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
    print_fn(f"================== {config.name} end ========================")

