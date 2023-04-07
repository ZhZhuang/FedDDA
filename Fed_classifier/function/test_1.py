import torch
import numpy as np

# # t = torch.Tensor([1,3,5])
# # t = t.tolist()
# # t += []
# # print(t)
#
# t = torch.Tensor([1,3,5])
# print(t.mean())


# pseudo_label = torch.randn(size=(5,10))
# class_acc = torch.Tensor([0.5]*10)
# p_cutoff = 0.9
# max_probs, max_idx = torch.max(pseudo_label, dim=-1)
# # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
# # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
# print(max_idx)
# print(class_acc[max_idx])
# print((2. - class_acc[max_idx]))
# print(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx])))
# mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
# print(mask)

# 支持索引选择元素
# t = torch.Tensor([1, 3, 5])
# t2 = t[[0,2]]
# print(t2)

# t = ["hello", 3, 5]
# # t = np.ndarray
# t = np.array(t)     # ?
# # t2 = t[[0,2]]
# t2 = t[[0, 2]]
# print(type(t2),t2, type(t), t)

# # a = np.array(["hello", 3, 5])
# a = np.array([ 0 ,3, 5]).astype(str)
# print(a)
# print(np.delete(a,np.where(a == "3")))
#
# a = np.arange(12).reshape(3,4)
# print(a)
# print(np.where(a < 2)[0])
#
# t = [ 0 ,3, 5]
# t += [ 0 ,3, 5]
# print(t)


# numpy 数组拼接  方法 一
a = np.arange(12).reshape(3,4)
b = np.arange(12).reshape(3,4)
# print(a)
# t = np.append(a,b, axis= 0)
# print( t.shape )
# print( t )

# numpy 数组拼接  方法 二
t2 = np.concatenate((a,b),axis=0)
print(t2.shape)
print(t2)
# concatenate 效率更高 , ?

c = np.array()
a= a.reshape(12)
t3 = np.concatenate((a,c),axis=0)
print(t3.shape)
print(t3)