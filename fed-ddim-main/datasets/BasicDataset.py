import copy

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image # ？？？  查看官方文档

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 # is_ulb=False,
                 # strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        # self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        # self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        # if self.is_ulb:         # 无标签数据集
        #     if strong_transform is None:
        #         self.strong_transform = copy.deepcopy(transform)
        #         # 下面的看不懂 ！！！！！
        #         self.strong_transform.transforms.insert(0, RandAugment(3, 5)) # list的 insert方法，
        #                                                                     # 有点蒙当时， python基础不牢
        # else:
        #     self.strong_transform = strong_transform

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target  # ToTensor()(img) 声明类 ，然后调用
        if isinstance(img, np.ndarray):  # ? 判断是否为 np.ndarray 类型
            img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, idx      # ToTensor()(img) 声明类 ，然后调用

    def __len__(self):
        return len(self.data)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot