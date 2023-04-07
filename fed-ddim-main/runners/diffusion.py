import copy
import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def fed_train(self, clients ):
        start_epoch, step = 0, 0
        print_fn = logging.info
        round = self.config.fed.round
        num_client = len(clients)

        # 加载保存的参数
        if self.args.resume_training:
            print_fn("resume_training ... ...")
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            for client in clients:
                client.model.load_state_dict(states[0])
                # client.optimizer.load_state_dict(states[1])
            start_epoch = states[1] + 1
            step = states[2]

        w_locals = [clients[0].model.state_dict() for i in range(len(clients))]  # 也可以b = [0]*10
        for r in range( start_epoch, round ):
            print_fn("round: {} begain!".format(r))
            start_time = time.time()
            for index, client in enumerate( clients ):
                print_fn("round: {}, client: {}, doing ...".format(r, index))
                self.train(client, index, r)
                w_locals[index] = copy.deepcopy(client.model.state_dict())

            # 总数据量
            total_data_points = sum([clients[i].num_sample for i in range(num_client)])
            # 用户数据量的比例
            fed_avg_freqs = [clients[i].num_sample / total_data_points for i in range(num_client)]

            # 联邦聚合
            with torch.no_grad():
                w_global = FedAvg(w_locals, fed_avg_freqs)
                # w_optim_global = FedAvg(w_optims)
            # 加载
            for index in range(len(clients)):
                # clients[index].model.load_state_dict(w_global)
                clients[index].model.load_state_dict(w_global, strict=True)

            # 保存参数
            states = [w_global,  # 模型线束
                      r,
                      step,]
            torch.save(  states,
                        os.path.join(self.args.log_path, "ckpt.pth")
                    )
            if ( r + 1 ) % self.config.fed.save_interval == 0 :
                torch.save(
                    states,
                    os.path.join(self.args.log_path, "ckpt_round_{}.pth".format(r))
                )
            dur_time = time.time() - start_time
            print_fn("round: {} end, with {} sec".format(r, dur_time))


    def train(self, client , idx, r):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger   # 见过两次
        # dataset, test_dataset = get_dataset(args, config)
        train_loader = client.train_loader
        model = client.model

        model = model.to(self.device)
        # model = torch.nn.DataParallel(model)

        optimizer = client.optimizer

        start_epoch, step = 0, 0

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y, _) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                y = y.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                # t = torch.randint(0, self.num_timesteps, (n,), device=self.device).long()
                loss = loss_registry[config.model.type](model, x, t, y, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"clients: {idx}, round: {r}, local_epoch: {epoch}, step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                # 是否是联邦学习下 不收敛的原因？
                # try:
                #     torch.nn.utils.clip_grad_norm_(
                #         model.parameters(), config.optim.grad_clip
                #     )
                # except Exception:
                #     pass
                optimizer.step()

                data_start = time.time()
        # client.model = model.cpu()     # 放到内存中,一个model 1.1 GB ,仍然解决不了本地训练的问题, 只能放到主存中

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    # os.path.join(self.args.log_path, "ckpt.pth"),
                    os.path.join(self.args.log_path, "ckpt_round_399.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,    # ？
                )
            model = model.to(self.device)
            # model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            # model = torch.nn.DataParallel(model)    # ？

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        # 加
        elif self.args.generate:
            self.sample_generate(model, 12000, start_c= 0, end_c=10)    #  1w  Cifar10    1.2w FashionMNIST
            # self.sample_generate(model, 20000, start_c=0, end_c= 10)  # 1w  Cifar10
        else:
            raise NotImplementedError("Sample procedeure not defined")

    # 0,1 已生成
    def sample_generate(self,model , num = 1024, start_c = 0, end_c = 10):
        '''
        num : 每类样本的数量
        '''
        config = self.config
        # batch_size = 512    # 512 10G显存
        batch_size = 256  # 512 10G显存
        last_batch = num % batch_size
        n_round = 0
        if last_batch == 0 :
            n_round = num // batch_size
        else:
            n_round = num // batch_size + 1

        print("开始生成图片！")

        for c in range(start_c, end_c):
            print("第 {} 类开始生成...".format(c))
            # n_round = num // batch_size  # 因为一次不能生成太多图片
            temp = 0
            bs = 0
            data_start = time.time()
            for r in range(n_round):
                if last_batch != 0 and r + 1 == n_round :
                    bs = last_batch
                else:
                    bs = batch_size
                x = torch.randn(
                    bs,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                labels = [c for _ in range(bs)]
                y = torch.IntTensor(labels).cuda()  # 超窜上 pytorch 1.7需要用 long ，超算显卡 Tesla t4
                with torch.no_grad():
                    _, x = self.sample_image(x, y, model, last=False)   #
                x = [inverse_data_transform(config, i) for i in x]  # ?
                for j in range(x[-1].size(0)):
                    tvu.save_image(
                        x[-1][j], os.path.join(self.args.image_folder, f"{c}_{temp}.png")
                    )
                    temp += 1
                print("class {}, {} / {}".format(c, temp, num))
            dur = time.time() - data_start
            print("class {}, {} / {}, time:{} sec per class".format(c, temp, num, dur))
            print("第 {} 类生成完毕!!!!!！".format(c))


    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config
        num = 10
        x = torch.randn(
            num,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        # y = torch.randint(0, 10, (num,)).cuda()
        y = torch.IntTensor(range(10)).cuda()

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, y, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        # i step ， j 第 j个样本
        # for i in range(len(x)):
        #     for j in range(x[i].size(0)):
        #         tvu.save_image(
        #             x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
        #         )
        for j in range(x[-1].size(0)):
            tvu.save_image(
                x[-1][j], os.path.join(self.args.image_folder, f"{j}.png")
            )


    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, y, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":  # 默认
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, y, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass



def FedAvg(w, wl):
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

# def FedAvg(w):
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             #print('done')
#             w_avg[k] += w[i][k]     # 有没有不能汇聚的层 ？
#         # w_avg[k] = torch.div(w_avg[k], len(w))
#         w_avg[k] = torch.true_divide(w_avg[k], len(w))  #兼容pytorch 1.6
#     return w_avg