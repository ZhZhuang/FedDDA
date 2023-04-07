import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)  # !!?
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    # 每一类的标签放到class_idcs的对应元素中, np.argwhere函数: 返回  non-zero 元素的位置
    class_idcs = [np.argwhere(train_labels==y).flatten()
                                    for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def cifar_noniid_dirichlet( num_users,  alpha = 1.0 ):

    # dataset = CIFAR10(root="./data", train=True, download=False)
    # test_dataset = CIFAR10(root="./data", train=False, download=False)  # 1w 张
    path = "/home/user/PycharmProjects/data/cifar"
    dataset = datasets.CIFAR10(root=path,  download=False, train=True)
    test_dataset = datasets.CIFAR10(root=path,  download=False, train=False)


    train_labels = np.array(dataset.targets)
    client_idcs = dirichlet_split_noniid(train_labels, alpha, num_users)
    test_labels = np.array(test_dataset.targets)
    client_test_idcs = dirichlet_split_noniid(test_labels, alpha, num_users)   # 上下比例不一定相同,日后再改

    # 下面无缝连接
    data_dict_list = []
    test_data_dict_list = []
    targets = np.array(dataset.targets)  # 方便使用 numpy的索引操作
    test_targets = np.array(test_dataset.targets)  # 方便使用 numpy的索引操作
    for i in range(num_users):
        dict = {}
        test_dict = {}
        # dict["sub_data"] = dataset.data[dict_users[i]]  # numpy的[]操作
        dict["sub_data"] = dataset.data[ client_idcs[i] ]  # numpy的[]操作
        test_dict["sub_data"] = test_dataset.data[ client_test_idcs[i] ]  # numpy的[]操作
        # dict["sub_target"] = dataset.targets[dict_users[i]] # dataset.targets 是 list
        temp_targets = targets[client_idcs[i]]
        test_temp_targets = test_targets[ client_test_idcs[i] ]
        dict["sub_targets"] = temp_targets.tolist()
        test_dict["sub_targets"] = test_temp_targets.tolist()
        data_dict_list.append(dict)
        test_data_dict_list.append(test_dict)
    return data_dict_list, test_data_dict_list

if __name__ == "__main__":
    np.random.seed(42)

    N_CLIENTS = 10
    # DIRICHLET_ALPHA = 1.0     # 越大,越是均衡
    DIRICHLET_ALPHA = 0.1
    # train_data = datasets.EMNIST(root=".", split="byclass", download=True, train=True)
    # test_data = datasets.EMNIST(root=".", split="byclass", download=True, train=False)
    # path = '/home/user/PycharmProjects/data/fmnist'
    # train_data = datasets.FashionMNIST(root=path, download=False, train=True)
    # test_data = datasets.FashionMNIST(root=path,  download=False, train=False)
    path = "/home/user/PycharmProjects/data/cifar"
    train_data = datasets.CIFAR10(root=path,  download=False, train=True)
    test_data = datasets.CIFAR10(root=path,  download=False, train=False)
    # n_channels = 1

    # cifar_noniid_dirichlet(5, alpha= 1.0)     #

    input_sz, num_cls = train_data.data[0].shape[0],  len(train_data.classes)
    train_labels = np.array(train_data.targets)

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)


    # 展示不同client的不同label的数据分布
    # 直方图
    plt.figure(figsize=(20,3))
    plt.hist([train_labels[idc]for idc in client_idcs], stacked=True,
            bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
    plt.xticks(np.arange(num_cls), train_data.classes)
    plt.legend()
    plt.show()