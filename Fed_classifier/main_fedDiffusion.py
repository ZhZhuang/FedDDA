from common import init, load_models
from core_feddiffusion import fed_train
from utils import get_logger

# def load_model(logger):
#     n_client= 10 #10  # 5  20
#     # 准备数据
#     # iid
#     # train_data_dict_list = get_user_data(5, train=True, dataname="FashionMNIST")
#     # val_data_dict_list = get_user_data(5, train=False, dataname="FashionMNIST")
#
#     # path = "/home/user/PycharmProjects/data/cifar"
#     # train_data_dict_list, _ = cifar_noniid(n_client, path= path)
#
#     # 和上一种分布，使用统一生成数据的情况下基本相同 alpha = 0.5 ， 1.0
#     # non iid 2  Dirichlet分布
#     train_data_dict_list, _ = cifar_noniid_dirichlet(n_client, alpha= 1.0)
#
#
#     # 生成样本
#     # transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
#     # t_transform = transforms.Compose([transforms.Pad(4),
#     #                                 transforms.RandomHorizontalFlip(),  # ? 水平翻转
#     #                                 transforms.RandomCrop(32),
#     #                                 transforms.ToTensor(),
#     #                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     #                                 ]
#     #                                )
#     t_transform = transforms.Compose([
#                                       transforms.Pad(4),
#                                       transforms.RandomHorizontalFlip(),  # ? 水平翻转
#                                       transforms.RandomCrop(32),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#                                       ]
#                                      )
#
#     # 共 2**10 * 10 * 10 样本  扩散模型  150 epoch 5 user
#     # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_noniid_imgs_10w_5user_150epoch", transform=transform)
#     # data_list = torch.utils.data.random_split(t_dataset, [2**10 * 10 for _ in range(n_client)])
#     # data_list = torch.utils.data.random_split(t_dataset, [2 ** 10 * 10 for _ in range(10)])
#     # 5 user 使用全部数据 2 ** 10 * 10 * 10
#     # data_list = torch.utils.data.random_split(t_dataset, [2 ** 10 * 10 * 2 for _ in range(5)])
#
#
#     # Confusion model生成的样本
#     # t_dataset = fileDataset("/home/user/PycharmProjects/FedConfusion/data/generate_imgs", transform=transform)
#     # dataloader_list = [torch.utils.data.DataLoader(t_dataset, 64, shuffle=True) for _ in range(5)]
#
#     # # cifar10 扩散模型生成样本，350 epoch，单模型训练，用于测试
#     # t_dataset = fileDataset("/home/user/PycharmProjects/data/gen_data_diffusion/cifar_50000",
#     #                         transform=t_transform,
#     #                         is_gray= False,
#     #                         )
#
#
#
#     # path_data_file = "/home/user/Gen_Data/cifar10/cifar_5user_400round_10w"
#     # path_data_file = "/home/user/Gen_Data/cifar10/cifar10-10client-alpha1.0_10w"
#     # 测试 100 round 的生成图片
#     # path_data_file = "/home/user/Gen_Data/cifar10/cifar10-dirichlet-a-1.0-100round"
#
#     # GAN 100round 单模型图片，5000 * 10 张
#     path_data_file = "/home/user/PycharmProjects/ACGAN_cifar10-master/save"
#
#     # cifar10 扩散模型生成样本，400 epoch，5 用户，10w， 联邦训练的得到
#     t_dataset = fileDataset(path_data_file,
#                             transform=t_transform,
#                             is_gray=False,
#                             )
#     # 随机划分数据集
#     # data_list = torch.utils.data.random_split(t_dataset, [10000 for _ in range(5)])
#     # data_list = torch.utils.data.random_split(t_dataset, [10000 for _ in range(10)])
#     # data_list = torch.utils.data.random_split(t_dataset, [20000 for _ in range(5)])
#     dict_users = get_data_random_idx(t_dataset, 10)
#     # dict_users = get_data_random_idx(t_dataset, 5)
#     data_list = []
#     for i in range(n_client):
#         idxs = dict_users[i]
#         dataset = fileDataset(path_data_file,
#                             transform=t_transform,
#                             is_gray=False,
#                             idxs= idxs
#                             )
#         data_list.append(dataset)
#
#     # 初始化用户
#     client_list = []
#
#     for i in range(n_client):
#         # C_model = MLP()
#         # C_model = ResModel(in_ch= 3)
#         # wrn_builder = build_WideResNet(1, 28, 2, 0.001, 0.1, 0.0)  # 单模型 数据不Normalize 0.81
#         # C_model = wrn_builder.build(10)
#         # C_model = ResNet50(n_class= 10, is_remix=False, is_gray= False)
#         C_model = ResNet18()  # ResNet 18
#         data = train_data_dict_list[i]["sub_data"]
#         targets = train_data_dict_list[i]["sub_targets"]
#
#         ################# 获取每类样本的数量，并保存在文件中
#         count = [0 for _ in range(10)]
#         for c in targets:  # lb_targets 为 0 ～ 9 ， 有操作
#             count[c] += 1
#         g_count = [0 for _ in range(10)]
#         for c in data_list[i].labels:  #
#             g_count[c] += 1
#         out = {"distribution": count,
#                "generated_data_distribution": g_count}
#         output_file = "save/client_data_statistics_%d.json"%(i)
#         # if not os.path.exists(output_file):
#         #     os.makedirs(output_file, exist_ok=True)
#         with open(output_file, 'w') as w:
#             json.dump(out, w)
#         #################
#
#         # transform = transforms.Compose( [transforms.Resize(32), transforms.ToTensor()] )
#         transform = transforms.Compose([
#                                         transforms.Pad(4),
#                                         transforms.RandomHorizontalFlip(),  # ? 水平翻转
#                                         transforms.RandomCrop(32),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#                                         ]
#                                        )
#         dataset = BasicDataset(data, targets, transform=transform,onehot= False)
#
#         ## 只使用自身数据
#         # dataloader = torch.utils.data.DataLoader(dataset, opt.batch_size,
#         #                                          shuffle=True, )
#         # 混合生成样本和真实样本
#         dataloader = torch.utils.data.DataLoader(dataset + data_list[i], opt.batch_size,
#                                                  shuffle=True,
#                                                  num_workers= 4)
#
#         client = Client(C_model=C_model, client_idx=i,
#                         dataloader= dataloader,
#                         dataset = dataset,
#                         logger= logger)
#
#         # Optimizers
#         optimizer_C = torch.optim.SGD(C_model.parameters(),lr=0.001, momentum= 0.9)    # 0.01
#         # optimizer_C = torch.optim.Adam(C_model.parameters(), lr=0.001)  #更适合resnet18
#         client.set_optimizer( optimizer_C )
#         client_list.append( client )
#
#     return client_list , t_transform, data_list

# def init():
#     seed = 1234 # 0
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
#     torch.cuda.manual_seed_all(seed)
#
#     cudnn.deterministic = True  # 随机数种子seed确定时，模型的训练结果将始终保持一致
#     cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
#
#     # pass

def main():
    config = init()

    # 获取日志
    logger_level = "INFO"
    logger = get_logger("col",
                        config.save_dir,
                        logger_level,
                        f"log_{config.name}.txt")  # 收为己用

    model_list , tf , data_list = load_models(config,logger)

    fed_train(
                config,
                model_list,
                logger=logger,
                transform=tf,
                data_list= data_list)

if __name__ == "__main__":
    main()
