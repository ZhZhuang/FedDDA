import os

import numpy as np
import torch
from torchvision.utils import save_image

from models.simple_model import Generator


cuda = True
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

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

if __name__ == "__main__":
    # path = "save/generate_40.pth"
    path = "save-local-epoch-2/generate_99.pth"
    out_path = "data/generate_imgs"
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    G = Generator().cuda()
    state_dict= torch.load(path)
    G.load_state_dict(state_dict)
    # 生成图片的多样性是个问题
    # gen_image(n_imgs=1000, n_class=10, generator=G, path=out_path)
    gen_image(n_imgs=1280, n_class=10, generator=G, path=out_path)
    # gen_image(n_imgs=10, n_class=10, generator=G, path=out_path)