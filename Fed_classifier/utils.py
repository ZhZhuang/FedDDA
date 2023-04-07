import os
import time

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST


# '''
# 工具类：
# 重写参数
# 加载模型
# 获取数据集,iid,noniid
# 获取日志
# '''


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):   # 感觉高手对语言掌握的很熟练，很熟悉
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'

    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c': 5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")

# 核心函数
def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:   # network names in torchvision.models
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]

    else:   #这条路  net_from_name: False
        if net_name == 'WideResNet':    # WideResNet  使用默认参数
            import models.nets.wrn as net   # wide res net
            builder = getattr(net, 'build_WideResNet')()        #python 很灵活, 先使用默认值创建 对象
        elif net_name == 'WideResNetVar':
            import models.nets.wrn_var as net
            builder = getattr(net, 'build_WideResNetVar')()
        elif net_name == 'ResNet50':
            import models.nets.resnet50 as net
            builder = getattr(net, 'build_ResNet50')(is_remix)
        else:
            assert Exception("Not Implemented Error")

        if net_name != 'ResNet50':
            # 给对象的属性复制 ！！
            setattr_cls_from_kwargs(builder, net_conf)      # 吸收
        return builder.build


def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)


def get_logger(name, save_path=None, level='INFO',file_name = None):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    # 终端打印两次
    # streamHandler = logging.StreamHandler()
    # streamHandler.setFormatter(log_format)
    # logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        if file_name is None:
            fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        else:
            fileHandler = logging.FileHandler(os.path.join(save_path, file_name))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 问题很大，连续的数据 target 是一样的，cifar的数据就是这样存放的
# def get_public_data(args,num):
#     dataset = CIFAR10(root=args.data_dir, train = True,  download = False)
#     data = dataset.data[len(dataset)-1-num : len(dataset)-1]
#     # 可以用 randomsploit对数据集进行划分
#     return data

def get_public_data(path,num, dataname = "Cifar10",download = True):
    if dataname == "Cifar10":
        dataset = CIFAR10(root=path, train=True, download=download)
    elif dataname == "FashionMNIST":
        dataset = FashionMNIST("./data", train=True, download=download,transform=transforms.ToTensor())
    index = np.random.choice(range(len(dataset)), num)
    pub_data = dataset.data[index] # 细节 不是()
    if dataname == "FashionMNIST":
        pub_data = pub_data.numpy()     # FashionMNist 默认是tensor，Transformer操作不能是tensor
    # alignment_data, _ = torch.utils.data.random_split(public_data, [N_alignment, len(public_data)-N_alignment])
    return pub_data


# 思路二：每个用户10000张样本：四类，2500张每类
def cifar_noniid( num_users, path ="./data/cifar",
                  download = True):

    dataset = CIFAR10(root=path, train=True, download=download)
    test_dataset = CIFAR10(root=path, train=False, download=download)  # 1w 张
    # dict_users = {}

    # num_shards × num_imgs = 总样本数
    # num_shards, num_imgs = 25, 2000 # 思路一
    # test_num_imgs = 400    #思路一，train和test 共享num_shards

    if num_users == 5:
        num_shards, num_imgs = 20, 2500  # 思路二  20 * 2500 = 5 w
        test_num_imgs = 500  # 以为 500 × 20 = 1w ，测试集的样本数
    elif num_users == 10:
        num_shards, num_imgs = 40, 1250  # 40 * 1250 = 5 w
        test_num_imgs = 250  # 以为 250 × 40 = 1w ，测试集的样本数
    elif num_users == 20:
        num_shards, num_imgs = 80, 625  # 80 * 625 = 5 w
        test_num_imgs = 125  # 以为 125 × 80 = 1w ，测试集的样本数
    else:
        assert Exception(f"Not Implemented! {num_users} clients is not implemented!")

    idx_shard = [i for i in range(num_shards)]  # 理解为总样本划分的份数
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    test_idxs = np.arange(num_shards * test_num_imgs)
    # labels = dataset.targets.numpy()    # 属性与mnist不同
    labels = np.array(dataset.targets)  # 属性与mnist不同
    test_labels = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 垂直拼接，对应的hstack水平拼接， 都有相应的限制
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # argsot元素大小排序，输出由小到大的索引值
    idxs = idxs_labels[0, :]  # 把标签由小到大排序后的索引
    test_idxs_labels = np.vstack((test_idxs, test_labels))
    test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :].argsort()]
    test_idxs = test_idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        # rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # rand_set = set(np.random.choice(idx_shard, 5, replace=False)) #思路一
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))  # 思路二
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # rand*num_imgs:(rand+1)*num_imgs] 表示选取 长度为num_ings的一块
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            test_dict_users[i] = np.concatenate(
                (test_dict_users[i], test_idxs[rand * test_num_imgs:(rand + 1) * test_num_imgs]), axis=0)
    # return dict_users  # 获取每个用户的样本的索引

    # 下面无缝连接
    data_dict_list = []
    test_data_dict_list = []
    targets = np.array(dataset.targets)  # 方便使用 numpy的索引操作
    test_targets = np.array(test_dataset.targets)  # 方便使用 numpy的索引操作
    for i in range(num_users):
        dict = {}
        test_dict = {}
        dict["sub_data"] = dataset.data[dict_users[i]]  # numpy的[]操作
        test_dict["sub_data"] = test_dataset.data[test_dict_users[i]]  # numpy的[]操作
        # dict["sub_target"] = dataset.targets[dict_users[i]] # dataset.targets 是 list
        temp_targets = targets[dict_users[i]]
        test_temp_targets = test_targets[test_dict_users[i]]
        dict["sub_targets"] = temp_targets.tolist()
        test_dict["sub_targets"] = test_temp_targets.tolist()
        data_dict_list.append(dict)
        test_data_dict_list.append(test_dict)
    return data_dict_list, test_data_dict_list

# 10 用户 固定， 用户有用的类别不不固定
def cifar_noniid_byclass( num_class, path ="./data/cifar",
                  download = True):

    dataset = CIFAR10(root=path, train=True, download=download)
    test_dataset = CIFAR10(root=path, train=False, download=download)  # 1w 张
    # dict_users = {}

    # num_shards × num_imgs = 总样本数
    # num_shards, num_imgs = 25, 2000 # 思路一
    # test_num_imgs = 400    #思路一，train和test 共享num_shards

    num_users = 10

    if num_class == 2:
        num_shards, num_imgs = 20, 2500  # 10 * 2 块， 块大小 2500
        test_num_imgs = 500  # 以为 500 × 20 = 1w ，测试集的样本数
    elif num_class == 4:
        num_shards, num_imgs = 40, 1250  # 10 * 4 * 1250 = 5 w
        test_num_imgs = 250  # 以为 250 × 40 = 1w ，测试集的样本数
    # elif num_users == 20:
    #     num_shards, num_imgs = 80, 625  # 80 * 625 = 5 w
    #     test_num_imgs = 125  # 以为 125 × 80 = 1w ，测试集的样本数
    else:
        assert Exception(f"Not Implemented! {num_class} class is not implemented!")

    idx_shard = [i for i in range(num_shards)]  # 理解为总样本划分的份数
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    test_idxs = np.arange(num_shards * test_num_imgs)
    # labels = dataset.targets.numpy()    # 属性与mnist不同
    labels = np.array(dataset.targets)  # 属性与mnist不同
    test_labels = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 垂直拼接，对应的hstack水平拼接， 都有相应的限制
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # argsot元素大小排序，输出由小到大的索引值
    idxs = idxs_labels[0, :]  # 把标签由小到大排序后的索引
    test_idxs_labels = np.vstack((test_idxs, test_labels))
    test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :].argsort()]
    test_idxs = test_idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class, replace=False))
        # rand_set = set(np.random.choice(idx_shard, 4, replace=False))  # 4 类
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # rand*num_imgs:(rand+1)*num_imgs] 表示选取 长度为num_ings的一块
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            test_dict_users[i] = np.concatenate(
                (test_dict_users[i], test_idxs[rand * test_num_imgs:(rand + 1) * test_num_imgs]), axis=0)
    # return dict_users  # 获取每个用户的样本的索引

    # 下面无缝连接
    data_dict_list = []
    test_data_dict_list = []
    targets = np.array(dataset.targets)  # 方便使用 numpy的索引操作
    test_targets = np.array(test_dataset.targets)  # 方便使用 numpy的索引操作
    for i in range(num_users):
        dict = {}
        test_dict = {}
        dict["sub_data"] = dataset.data[dict_users[i]]  # numpy的[]操作
        test_dict["sub_data"] = test_dataset.data[test_dict_users[i]]  # numpy的[]操作
        # dict["sub_target"] = dataset.targets[dict_users[i]] # dataset.targets 是 list
        temp_targets = targets[dict_users[i]]
        test_temp_targets = test_targets[test_dict_users[i]]
        dict["sub_targets"] = temp_targets.tolist()
        test_dict["sub_targets"] = test_temp_targets.tolist()
        data_dict_list.append(dict)
        test_data_dict_list.append(test_dict)
    return data_dict_list, test_data_dict_list

# fashion mnist 训练样本：60000,测试样本：10000,图片尺寸：28x28,单通道
def fmnist_noniid( num_users, path = "./data",
                   download = True):
    # Fashion-MNIST为1x28x28,使用Densenet时需resize为 96x96
    #ransforms.Compose([transforms.Resize(size=96), transforms.ToTensor()])
    dataset = FashionMNIST( path, train=True, download=download, transform=transforms.ToTensor())
    test_dataset = FashionMNIST( path, train=False, download=download, transform=transforms.ToTensor()) # 1w 张
    # dict_users = {}

    # 5 user
    if num_users == 5:
        num_shards, num_imgs = 20, 3000  # 思路二  20 x 3000 = 6w
        test_num_imgs = 500  # 以为 500 × 20 = 1w ，测试集的样本数

    # 10
    elif num_users ==10:
        num_shards, num_imgs = 40, 1500  # 思路二  40 x 1500 = 6w
        test_num_imgs = 250  # 因为 250 × 40 = 1w ，测试集的样本数

    # 20
    elif num_users == 20:
        num_shards, num_imgs = 80, 750  # 思路二  80 x 750 = 6w
        test_num_imgs = 125  # 因为 125 × 80 = 1w ，测试集的样本数

    else:
        assert Exception(f"Not Implemented! {num_users} clients is not implemented!")

    idx_shard = [i for i in range(num_shards)]  # 理解为总样本划分的份数
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    test_idxs = np.arange(num_shards * test_num_imgs)
    # labels = dataset.targets.numpy()    # 属性与mnist不同
    labels = np.array(dataset.targets)  # 属性与mnist不同
    test_labels = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 垂直拼接，对应的hstack水平拼接， 都有相应的限制
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # argsot元素大小排序，输出由小到大的索引值
    idxs = idxs_labels[0, :]  # 把标签由小到大排序后的索引
    test_idxs_labels = np.vstack((test_idxs, test_labels))
    test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :].argsort()]
    test_idxs = test_idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        # rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # rand_set = set(np.random.choice(idx_shard, 5, replace=False)) #思路一
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))  # 思路二
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # rand*num_imgs:(rand+1)*num_imgs] 表示选取 长度为num_ings的一块
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            test_dict_users[i] = np.concatenate(
                (test_dict_users[i], test_idxs[rand * test_num_imgs:(rand + 1) * test_num_imgs]), axis=0)
    # return dict_users  # 获取每个用户的样本的索引

    # 下面无缝连接
    data_dict_list = []
    test_data_dict_list = []
    targets = np.array(dataset.targets)  # 方便使用 numpy的索引操作
    test_targets = np.array(test_dataset.targets)  # 方便使用 numpy的索引操作
    for i in range(num_users):
        dict = {}
        test_dict = {}
        dict["sub_data"] = dataset.data[dict_users[i]].numpy()  #  fashion mnist 默认的数据是tensor，需要转为numpy，方便后面操作
        test_dict["sub_data"] = test_dataset.data[test_dict_users[i]].numpy()  #
        temp_targets = targets[dict_users[i]]
        test_temp_targets = test_targets[test_dict_users[i]]
        dict["sub_targets"] = temp_targets.tolist()
        test_dict["sub_targets"] = test_temp_targets.tolist()
        data_dict_list.append(dict)
        test_data_dict_list.append(test_dict)
    return data_dict_list, test_data_dict_list

# 用户数量固定 = 10
def fmnist_noniid_byclass( num_class, path = "./data",
                   download = True):
    """
    :param num_class: 用户包含的样本类型总数，支持 2，4
    :param path:
    :param download:
    :return:
    """
    # Fashion-MNIST为1x28x28,使用Densenet时需resize为 96x96
    #ransforms.Compose([transforms.Resize(size=96), transforms.ToTensor()])
    dataset = FashionMNIST( path, train=True, download=download, transform=transforms.ToTensor())
    test_dataset = FashionMNIST( path, train=False, download=download, transform=transforms.ToTensor()) # 1w 张
    # dict_users = {}

    num_users = 10

    # 5 user
    if num_class == 2:
        num_shards, num_imgs = 20, 3000  # 思路二  10 * 2 * 3000 = 20 x 3000 = 6w
        test_num_imgs = 500  # 以为 500 × 20 = 1w ，测试集的样本数

    # 10
    elif num_class == 4:
        num_shards, num_imgs = 40, 1500  # 思路二  40 x 1500 = 6w
        test_num_imgs = 250  # 因为 250 × 40 = 1w ，测试集的样本数

    else:
        assert Exception(f"Not Implemented! {num_class} class is not implemented!")

    idx_shard = [i for i in range(num_shards)]  # 理解为总样本划分的份数
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    test_idxs = np.arange(num_shards * test_num_imgs)
    # labels = dataset.targets.numpy()    # 属性与mnist不同
    labels = np.array(dataset.targets)  # 属性与mnist不同
    test_labels = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 垂直拼接，对应的hstack水平拼接， 都有相应的限制
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # argsot元素大小排序，输出由小到大的索引值
    idxs = idxs_labels[0, :]  # 把标签由小到大排序后的索引
    test_idxs_labels = np.vstack((test_idxs, test_labels))
    test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :].argsort()]
    test_idxs = test_idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class, replace=False))   # 选几块，几类数据
        # rand_set = set(np.random.choice(idx_shard, 4, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # rand*num_imgs:(rand+1)*num_imgs] 表示选取 长度为num_ings的一块
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            test_dict_users[i] = np.concatenate(
                (test_dict_users[i], test_idxs[rand * test_num_imgs:(rand + 1) * test_num_imgs]), axis=0)
    # return dict_users  # 获取每个用户的样本的索引

    # 下面无缝连接
    data_dict_list = []
    test_data_dict_list = []
    targets = np.array(dataset.targets)  # 方便使用 numpy的索引操作
    test_targets = np.array(test_dataset.targets)  # 方便使用 numpy的索引操作
    for i in range(num_users):
        dict = {}
        test_dict = {}
        dict["sub_data"] = dataset.data[dict_users[i]].numpy()  #  fashion mnist 默认的数据是tensor，需要转为numpy，方便后面操作
        test_dict["sub_data"] = test_dataset.data[test_dict_users[i]].numpy()  #
        temp_targets = targets[dict_users[i]]
        test_temp_targets = test_targets[test_dict_users[i]]
        dict["sub_targets"] = temp_targets.tolist()
        test_dict["sub_targets"] = test_temp_targets.tolist()
        data_dict_list.append(dict)
        test_data_dict_list.append(test_dict)
    return data_dict_list, test_data_dict_list

# 独立同分布划分 iid
# 将数据集 划分
def get_user_data( num_users, train = True, num_list=None,
                   dataname = 'CIFAR10',
                   path = "./data",
                   download = True):
    if dataname == 'CIFAR10':
        dataset = CIFAR10(root="./data", train=train, download=download)
    elif dataname == "FashionMNIST":
        # FashionMNIST继承了Dataset基类，此时dataset对象并没有执行transform中的变换，后面
        # 又将dataset对象中的数据单独拿出，所以时原始的数据 1*28*28
        dataset = FashionMNIST(
            path,
            train=True,
            download=download,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor()]
            ),
        )
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    num_items = [int(len(dataset) / num_users) for i in range(num_users)]

    if num_list!=None:
        # print(sum(num_list) ,len(dataset), len(num_list))
        assert sum(num_list) == len(dataset) and len(num_list)==num_users,'列表之和必须等于数据集的样本数！,列表长度必须等于num_users!'
        for i, temp in enumerate( num_list):
            num_items[i] = temp

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = sorted(list(dict_users[i])) # 排序
    # return dict_users
    data_dict_list = []
    targets = np.array(dataset.targets)     # 方便使用 numpy的索引操作
    for i in range(num_users):
        dict = {}
        # 注意：cifar中的data 是numpy，而FashionMNist中的data是 tensor，因此，统一转化为numpy
        dict["sub_data"] = dataset.data[dict_users[i]]  # numpy的[]操作
        if dataname == "FashionMNIST":
            dict["sub_data"] = dict["sub_data"].numpy()
        # dict["sub_target"] = dataset.targets[dict_users[i]] # dataset.targets 是 list
        temp_targets = targets[dict_users[i]]
        dict["sub_targets"] = temp_targets.tolist()
        data_dict_list.append(dict)
    return data_dict_list


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    # 每一类的标签放到class_idcs的对应元素中, np.argwhere函数 ?
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


def cifar_noniid_dirichlet( num_users, alpha = 1.0, path ="./data", download = True  ):

    # dataset = CIFAR10(root="./data", train=True, download=False)
    # test_dataset = CIFAR10(root="./data", train=False, download=False)  # 1w 张
    # path = "/home/user/PycharmProjects/data/cifar"
    dataset = CIFAR10(root=path,  download=download, train=True)
    test_dataset = CIFAR10(root=path,  download=download, train=False)


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

# fashion mnist 训练样本：60000,测试样本：10000,图片尺寸：28x28,单通道

def fmnist_noniid_dirichlet( num_users, alpha = 1.0, path = "./data" , download = True):
    # Fashion-MNIST为1x28x28,使用Densenet时需resize为 96x96
    #ransforms.Compose([transforms.Resize(size=96), transforms.ToTensor()])
    dataset = FashionMNIST( path, train=True, download=download, transform=transforms.ToTensor())
    test_dataset = FashionMNIST( path, train=False, download=download, transform=transforms.ToTensor()) # 1w 张
    # dict_users = {}

    train_labels = np.array(dataset.targets)
    client_idcs = dirichlet_split_noniid(train_labels, alpha, num_users)
    test_labels = np.array(test_dataset.targets)
    client_test_idcs = dirichlet_split_noniid(test_labels, alpha, num_users)  # 上下比例不一定相同,日后再改

    # 下面无缝连接
    data_dict_list = []
    test_data_dict_list = []
    targets = np.array(dataset.targets)  # 方便使用 numpy的索引操作
    test_targets = np.array(test_dataset.targets)  # 方便使用 numpy的索引操作
    for i in range(num_users):
        dict = {}
        test_dict = {}
        # dict["sub_data"] = dataset.data[dict_users[i]]  # numpy的[]操作
        dict["sub_data"] = dataset.data[client_idcs[i]].numpy()  # numpy的[]操作
        test_dict["sub_data"] = test_dataset.data[client_test_idcs[i]].numpy()  # numpy的[]操作
        # dict["sub_target"] = dataset.targets[dict_users[i]] # dataset.targets 是 list
        temp_targets = targets[client_idcs[i]]
        test_temp_targets = test_targets[client_test_idcs[i]]
        dict["sub_targets"] = temp_targets.tolist()
        test_dict["sub_targets"] = test_temp_targets.tolist()
        data_dict_list.append(dict)
        test_data_dict_list.append(test_dict)

    return data_dict_list, test_data_dict_list

def draw_picture( value_list, show = False):
    num = len(value_list)
    # x = []
    plt.plot(value_list,label=u'Acc')
    plt.xlabel(u"Communication Rounds")  # X轴标签
    plt.ylabel("Average Test Accuracy")  # Y轴标签
    # plt.title("net:WideResNet")  # 标题，中文无法输出
    plt.show()

# 原名 def split(dataset, num_users):
# 作用：将数据集随机划分
def get_data_random_idx(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])     # set -> list
    return dict_users


# 相对合理的划分数据集
# 思路：
#   1，计算每个client的样本总数，
#   2，获得总体一共多少类(10)
#   3，样本总数除以总类别数，获得每类样本的平均数
#   4，平均数减去每类样本数，若果大于零，则获取相应的样本，以达到平衡
# 所需参数：
# 生成样本，用户数据分布，用户数量
def get_lackdata_idx(dataset, dtbs, num_users,div = 10, n_classes=10, logger = None):
    """
    :param dataset: 生成数据的数据集
    :param dtbs: 每个用户的数据类别分布情况, 二维数组
    :param num_users: 用户数量，有点多余
    :return:
    """
    # 每一类的标签对应的位置，放到class_idcs的对应元素中

    print_fn = print if logger is None else logger.info

    labels = np.array(dataset.labels)
    class_idcs = [np.argwhere(labels==y).flatten().tolist()
                                    for y in range(n_classes)]
    dict_users = {i:[] for i in range(num_users)}
    for i in range(num_users):
        mean = sum(dtbs[i]) // div
        # mean = sum(dtbs[i]) // 4
        for j in range(n_classes):
            lack = mean - dtbs[i][j]    # j 样本缺少的数量
            if lack > 0 :
                class_n_idcs = class_idcs[j]
                # if sum(class_n_idcs) < lack:   # 剩余的没有缺少的样本多时，暂时的策略就不要了,犯糊涂 sum
                if len(class_n_idcs) < lack:
                    print_fn("用户: {},第 {} 类数据不足! 需要: {}, 剩余: {}".format(i, j, lack, len(class_n_idcs)))
                    lack = len(class_n_idcs)
                    # continue
                # print("classd_n_idcs:", len(class_n_idcs))
                # print("lack:", lack)
                n_sample = set(np.random.choice(class_n_idcs, lack, replace=False))
                class_n_idcs = list(set(class_n_idcs) - n_sample)   # j样本的剩余数量
                class_idcs[j] = class_n_idcs
                # dict_users[i].append(list(n_sample))
                dict_users[i] += (list(n_sample))
    return dict_users


    #
    # num_items = int(len(dataset) / num_users)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    #     all_idxs = list(set(all_idxs) - dict_users[i])
    #     dict_users[i] = list(dict_users[i])     # set -> list
    #
    # return dict_users