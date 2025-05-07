# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import print_function
import argparse
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
import time
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 添加项目根目录到路径中
from utils import load_features

# 创建参数解析器对象，并设置描述信息
parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')

# 添加参数：数据目录，必填项，字符串类型
parser.add_argument('--data-dir', type=str, required=True, help='data directory')

# 添加参数：结果目录，字符串类型，默认值为'result'
parser.add_argument('--result-dir', type=str, default='save/result', help='directory for saving results')
# 添加参数：特征提取器类型，字符串类型，默认值为'resnet50'
parser.add_argument('--extractor', type=str, default='resnet50', help='extractor type')
# 添加参数：数据集类型，字符串类型，默认值为'SST'
parser.add_argument('--dataset', type=str, default='SST', help='dataset')
# 添加参数：L2正则化参数，浮点数类型，默认值为1e-6
parser.add_argument('--lam', type=float, default=1e-6, help='L2 regularization')
# 添加参数：目标扰动的标准差，浮点数类型，默认值为10.0
parser.add_argument('--std', type=float, default=10.0, help='standard deviation for objective perturbation')
# 添加参数：要移除的数据点数量，整数类型，默认值为1000
parser.add_argument('--num-removes', type=int, default=10000, help='number of data points to remove')
# 添加参数：训练数据拆分数量，整数类型，默认值为1
parser.add_argument('--train-splits', type=int, default=1, help='number of training data splits')
# 添加参数：负样本子采样比率，浮点数类型，默认值为1.0
parser.add_argument('--subsample-ratio', type=float, default=1.0, help='negative example subsample ratio')
# 添加参数：优化步骤数量，整数类型，默认值为100
parser.add_argument('--num-steps', type=int, default=100, help='number of optimization steps')
# 添加参数：训练模式，字符串类型，默认值为'binary'
parser.add_argument('--train-mode', type=str, default='ovr', help='train mode [ovr/binary]')
# 添加参数：是否分别训练二分类器，布尔类型，默认值为False
parser.add_argument('--train-sep', action='store_true', default=False, help='train binary classifiers separately')
# 添加参数：是否在优化器中显示详细信息，布尔类型，默认值为False
parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')

args = parser.parse_args()

device = torch.device("cuda")


def lr_loss(w, X, y, lam):
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2
# 计算线性模型的损失函数，这里使用逻辑回归的对数似然损失函数，并加入L2正则化。

def lr_eval(w, X, y):
    return X.mv(w).sign().eq(y).float().mean()
# lr_eval：用于评估模型的准确率，返回预测标签与真实标签的匹配度。

def lr_grad(w, X, y, lam):
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z - 1) * y) + lam * X.size(0) * w
# lr_grad计算逻辑回归模型的梯度，用于优化模型

def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()
# lr_hessian_inv计算逻辑回归模型的Hessian矩阵的逆，用于在移除数据时进行模型参数的更新。

def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    # 1. 初始化参数 w
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)

    # 2. 定义闭包函数 closure，用于重新计算损失.
    #  返回损失函数的值。L-BFGS 优化器在每次更新时会调用这个闭包函数，以重新计算梯度和损失。
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam) # 不包含偏置项的损失
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)

    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data
#lr_optimize：使用L-BFGS算法优化逻辑回归模型的参数，逐步减少损失函数，支持添加扰动向量 b

def ovr_lr_loss(w, X, y, lam, weight=None):
    z = batch_multiply(X, w).mul_(y)
    if weight is None:

        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2
# 计算线性模型的损失函数，这里使用逻辑回归的对数似然损失函数，并加入L2正则化。

def ovr_lr_optimize(X, y, lam, weight=None, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(device), requires_grad=True)

    def closure():
        if b is None:
            return ovr_lr_loss(w, X, y, lam, weight)
        else:
            return ovr_lr_loss(w, X, y, lam, weight) + (b * w).sum() / X.size(0)

    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-10)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight)
        if b is not None:
            if weight is None:
                loss += (b * w).sum() / X.size(0)
            else:
                loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data

#batch_multiply：用于大规模矩阵相乘，按批次处理数据以节省内存
def batch_multiply(A, B, batch_size=500000):
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i + 1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(device)


def spectral_norm(A, num_iters=20):
    x = torch.randn(A.size(0)).float().to(device)
    norm = 1
    for i in range(num_iters):
        x = A.mv(x)
        norm = x.norm()
        x /= norm
    return math.sqrt(norm)
#spectral_norm：计算矩阵的谱范数，用于衡量 Hessian 矩阵的变化。

# 加载数据集的特征和标签
X_train, X_test, y_train, y_train_onehot, y_test = load_features(args)
X_test = X_test.float().to(device)
y_test = y_test.to(device)

save_path = '%s/%s_%s_splits_%d_ratio_%.2f_std_%.1f_lam_%.0e.pth' % (
    args.result_dir, args.extractor, args.dataset, args.train_splits, args.subsample_ratio, args.std, args.lam)


if os.path.exists(save_path):  # 检查模型是否已经存在
    # 加载已经训练好的模型参数
    checkpoint = torch.load(save_path)
    w = checkpoint['w']  # 权重参数
    b = checkpoint['b']  # 偏置参数
    weight = checkpoint['weight']  # 样本的权重
else:
    # 训练一个启用移除机制的线性模型
    start = time.time()  # 记录开始时间
    if args.subsample_ratio < 1.0:
        # 如果样本子集比例小于1，则针对负样本进行子采样以平衡正负样本比例
        subsample_indices = torch.rand(y_train_onehot.size()).lt(args.subsample_ratio).float()
        # 计算子样本的权重
        weight = (subsample_indices + y_train_onehot.gt(0).float()).gt(0).float()
        weight = weight / weight.sum(0).unsqueeze(0)  # 归一化权重
        weight = weight.to(device)  # 将权重转移到GPU设备
    else:
        weight = None  # 如果子集比例为1，权重设置为空
    # sample objective perturbation vector
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    y_train_onehot = y_train_onehot.float().to(device)
    if args.train_mode == 'ovr':
        # 初始化偏置b，使用高斯分布随机初始化
        b = args.std * torch.randn(X_train.size(1), y_train_onehot.size(1)).float().to(device)
        if args.train_sep:
            # train K binary LR models separately
            # 初始化权重矩阵w
            w = torch.zeros(b.size()).float().to(device)
            for k in range(y_train_onehot.size(1)):
                if weight is None:
                    w[:, k] = lr_optimize(X_train, y_train_onehot[:, k], args.lam, b=b[:, k], num_steps=args.num_steps,
                                          verbose=args.verbose)# 遍历每个类别，训练二分类模型
                else:
                    # 如果有权重，基于权重训练逻辑回归模型
                    w[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], y_train_onehot[:, k][weight[:, k].gt(0)],
                                          args.lam, b=b[:, k], num_steps=args.num_steps, verbose=args.verbose)
        else:
            # # 一次性联合训练所有二分类模型
            w= ovr_lr_optimize(X_train, y_train_onehot, args.lam, weight, b=b, num_steps=args.num_steps,
                                verbose=args.verbose)
    else:
        # 如果训练模式为二分类或其他模式，则训练一个二分类逻辑回归模型
        # 随机初始化权重向量b
        b = args.std * torch.randn(X_train.size(1)).float().to(device)
        w = lr_optimize(X_train, y_train, args.lam, b=b, num_steps=args.num_steps, verbose=args.verbose)
    print('Time elapsed: %.2fs' % (time.time() - start))
    torch.save({'w': w, 'b': b, 'weight': weight}, save_path)

if args.train_mode == 'ovr':
    pred = X_test.mm(w).max(1)[1]
    print('Test accuracy = %.4f' % pred.eq(y_test).float().mean())
else:
    pred = X_test.mv(w)
    print('Test accuracy = %.4f' % pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean())
# # 初始化梯度范数近似值数组和时间数组，用于记录每次移除样本后的梯度变化
grad_norm_approx = torch.zeros(args.num_removes).float()
times = torch.zeros(args.num_removes)
# 如果训练模式为“一对其余”，调整训练标签
if args.train_mode == 'ovr':
    y_train = y_train_onehot
# 复制当前权重矩阵，以进行移除样本时的近似更新
w_approx = w.clone()
# 随机排列索引
torch.manual_seed(42)  # 设置种子为42
torch.cuda.manual_seed_all(42)
# 随机排列索引
perm = torch.randperm(X_train.size(0)).to(y_train.device)
X_train = X_train.index_select(0, perm)
X_train = X_train.float().to(device)
y_train = y_train[perm].float().to(device)

# initialize K = X^T * X for fast computation of spectral norm
print('Preparing for removal')
if weight is None:
    K = X_train.t().mm(X_train) ## 计算X^T * X矩阵
else:
    # 如果指定了权重，根据权重计算多个K矩阵
    weight = weight.index_select(0, perm.to(device))
    Ks = []
    for i in range(y_train_onehot.size(1)):
        # 针对每个类别，计算对应的K矩阵
        X_sub = X_train.cpu()[weight[:, i].gt(0).cpu()]#选择当前类别中权重大于0的训练样本
        Ks.append(X_sub.t().mm(X_sub).to(device))#计算选定样本的转置矩阵与自身的矩阵乘积，并将结果添加到Ks列表中


print('Testing removal')

# 筛选 5000 条样本（保证移除样本来自这里）
num_samples_to_select = 5000
subsample_indices = torch.arange(num_samples_to_select)
X_train_subset = X_train[subsample_indices]
y_train_subset = y_train[subsample_indices]
remaining_indices = torch.arange(num_samples_to_select, len(X_train))
original_indices = torch.cat([subsample_indices, remaining_indices])

# 计算影响量
impact_norms = torch.zeros(len(X_train_subset))  # 记录每个样本的影响量
for i in range(len(X_train_subset)):
    impact_sum = 0  # 记录该样本在所有类别的影响量
    for k in range(y_train_onehot.size(1)):  # 对 K 个类别分别计算
        grad_i = lr_grad(w_approx[:, k], X_train[i].unsqueeze(0), y_train[i, k].unsqueeze(0),
                         args.lam)  # 计算当前样本的梯度
        # 计算剩余样本的 Hessian 逆矩阵（基于剩余的样本）
        X_rem = X_train[(i + 1):]  # 剩余训练样本的特征矩阵
        y_rem = y_train[(i + 1):, k]  # 剩余训练样本的标签
        H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, args.lam)
        Delta = H_inv.mv(grad_i)  # 计算影响量
        w_approx[:, k] += Delta  # 更新模型权重
        impact_sum += Delta.norm(p=2).item()  # 计算影响量范数，并更新影响量累积
    impact_norms[i] = impact_sum  # 存储每个样本的影响量
    print(f"Sample {i} (Original index: {original_indices[i].item()}) - impact_norms: {impact_norms[i]}")

# 根据影响量筛选 1000 个非典型样本
sorted_indices = torch.argsort(impact_norms, descending=True)
non_typical_samples = original_indices[sorted_indices[:1000]]  # 影响量最大的 1000 个样本
typical_samples = original_indices[sorted_indices[1000:]]  # 其余 4000 个样本

# 保存这些索引，用于后续操作
torch.save(non_typical_samples, 'robust\\non_typical_samples_sst.pth')
torch.save(typical_samples, 'robust\\typical_samples_sst.pth')

# 打印信息确认
print(f"Non-typical samples (top 1000) saved to 'non_typical_samples.pth'")
print(f"Typical samples (remaining 4000) saved to 'typical_samples.pth'")



