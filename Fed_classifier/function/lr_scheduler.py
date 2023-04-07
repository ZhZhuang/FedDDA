# //查看CUDA 的版本
# echo $CUDA_HOME
#
# //查看torch的版本

# import torch
# import torchvision
# print("torch.__version__: " ,  torch.__version__ ) #//注意是两个下划线
# print("torch.version.cuda:" , torch.version.cuda)  #//查看对应的CUDA版本
# print("torchvision.__version__:",torchvision.__version__)
#
# print("torch.cuda.is_available()", torch.cuda.is_available())

# torch.__version__:  1.7.1
# torch.version.cuda: 10.2
# torchvision.__version__: 0.8.2

# pip uninstall torch #删除原本的torch
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# torch.__version__:  1.10.1
# torch.version.cuda: 11.3
# torchvision.__version__: 0.8.2

# if __name__ == "__main__":
#     import torch
#     import matplotlib.pyplot as plt
#
#     model = torch.nn.Linear(2, 1)
#     optimizer = torch.optim.SGD(model.parameters(), lr=100)
#     lambda1 = lambda epoch: 0.65 ** epoch
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
#
#     lrs = []
#
#     for i in range(10):
#         optimizer.step()
#         lrs.append(optimizer.param_groups[0]["lr"])
#         #     print("Factor = ", round(0.65 ** i,3)," , Learning Rate = ",round(optimizer.param_groups[0]["lr"],3))
#         scheduler.step()
#
#     plt.plot(range(10), lrs)
#     plt.show()
#
# import torch
# import matplotlib.pyplot as plt
# model = torch.nn.Linear(2, 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=100)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)#T_max表示半个周期的大小
# lrs = []
#
#
# for i in range(100):
#     optimizer.step()
#     lrs.append(optimizer.param_groups[0]["lr"])
# #     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
#     scheduler.step()
# plt.plot(lrs)
# plt.show()

#
################  测试调度器
import math

from torch.optim.lr_scheduler import LambdaLR
#
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1000)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 1000)#T_max表示半个周期的大小
    lrs = []

    for i in range(1000):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    #     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
        scheduler.step()
    
        # scheduler.step()

    # scheduler.step(i) 将学习率直接调到，第 i 步
    # for i, lr in enumerate(lrs):
    #     scheduler.step(i)
    #     t = optimizer.param_groups[0]["lr"]
    #     print(lr == t)

    print(optimizer.param_groups[0]["lr"])
    plt.plot(lrs)
    plt.show()
    # # ################  测试调度器 end  ################
