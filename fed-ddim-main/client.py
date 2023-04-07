from torch.utils import data

from functions import get_optimizer
from models.diffusion import Model


class Client:
    def __init__(self, config , dataset ):
        # Cifar10 的模型占了 1.1GB, 无论是内存还是显存放不下,
        # 设计缓存池类似的粗略
        # 或者只声明 一个 model, 关键字保存到硬盘中,从硬盘中加载参数,但是降低速度
        self.model = Model(config)
        self.dataset = dataset
        self.train_loader = data.DataLoader(
                                dataset,
                                batch_size=config.training.batch_size,
                                shuffle=True,
                                num_workers=config.data.num_workers,
                            )
        self.optimizer = get_optimizer(config, self.model.parameters())

        self.num_sample = len(dataset)  # 样本数量

