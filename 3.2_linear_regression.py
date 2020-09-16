import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
from d2lzh_pytorch.utils import *

num_inputs = 2       # 特征数
num_examples = 1000  # 训练数据集样本数
true_w = [2, -3.4]   # 权重
true_b = 4.2         # 偏差
batch_size = 10      # 批量大小

# 生成数据集
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)  # 创建1000行2列标准正态分布Tensor
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 生成标签 y=Xw+b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)  # 标签加上噪声项 y=Xw+b+ϵ
# 噪声项 ϵ 服从均值为0、标准差为0.01的正态分布。噪声代表了数据集中无意义的干扰。
# features的每一行是一个长度为2的向量，而labels的每一行是一个长度为1的向量（标量）
# print(features[0], labels[0])

# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# # 第二个特征features[:, 1]和标签 labels 的散点图
# plt.show()

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break  # 输出第一批

# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# 之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们要让它们的requires_grad=True
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
