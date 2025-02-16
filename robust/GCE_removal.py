
from __future__ import print_function
import argparse
import math
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
import time
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
parser.add_argument('--q', type=float, default=1.0, help='q')
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

def lr_gce_grad(w, X, y, lam, q=0.7):
    s = y * X.mv(w)
    p = torch.sigmoid(s)
    # GCE 损失的一阶导数 d(loss)/ds = - p^q * (1-p)
    grad_loss = - p.pow(q) * (1 - p)
    grad = X.t().mv(grad_loss * y) + lam * X.size(0) * w
    return grad


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

def lr_gce_hessian_inv(w, X, y, lam, q=0.7, batch_size=50000):
    s = y * X.mv(w)
    p = torch.sigmoid(s)
    # 二阶导数：d²(loss)/ds² = - p^q * (1-p) * (q*(1-p) - p)
    d2 = - p.pow(q) * (1 - p) * (q * (1 - p) - p)
    H = None
    num_batch = int(math.ceil(X.size(0) / float(batch_size)))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        D_i = d2[lower:upper].unsqueeze(1)  # 将每个样本的二阶导数变成列向量
        if H is None:
            H = X_i.t().mm(D_i * X_i)
        else:
            H += X_i.t().mm(D_i * X_i)
    H_reg = H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)
    return H_reg.inverse()

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


def ovr_gce_loss(w, X, y, lam, weight=None, q=0.7):
    z = batch_multiply(X, w).mul_(y)
    p = torch.sigmoid(z)
    loss_term = (1 - p.pow(q)) / q  # GCE 损失
    if weight is None:
        loss = loss_term.mean(0).sum()
    else:
        loss = (loss_term * weight).sum()
    return loss + lam * w.pow(2).sum() / 2


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

def ovr_gce_optimize(X, y, lam, weight=None, b=None, num_steps=100, tol=1e-10, q=0.7, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return ovr_gce_loss(w, X, y, lam, weight, q=q)
        else:
            return ovr_gce_loss(w, X, y, lam, weight, q=q) + (b * w).sum() / X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-10)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_gce_loss(w, X, y, lam, weight, q=q)
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

save_path = '%s/%s_%s_splits_%d_ratio_%.2f_std_%.1f_lam_%.0e_H.pth' % (
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
            w = ovr_gce_optimize(X_train, y_train_onehot, args.lam, weight, b=b, num_steps=500, verbose=args.verbose, q=args.q)

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

# 加载之前保存的非典型样本和典型样本索引
non_typical_samples = torch.load('robust/non_typical_samples.pth')
X_train_first_5000 = X_train[:5000]
non_typical_indices = set(non_typical_samples.tolist())
typical_indices = [i for i in range(5000) if i not in non_typical_indices]
# typical_samples = X_train[typical_indices]
# 设置随机种子，确保每次运行一致
random.seed(32)
torch.manual_seed(32)
num_samples_to_sample = 2000

# 确定采样的比例：可以根据需要设置为 1.0:0, 0.7:0.3, 或 0.3:0.7
# ratio = [1.0, 0]  # 比如选择 1.0:0（完全选 typical_samples）
ratio = [0.8, 0.2]

# 根据比例计算每种类型的样本数量
num_typical = int(num_samples_to_sample * ratio[0])
num_not_typical = num_samples_to_sample - num_typical

# 从 non_typical_samples 和 typical_samples 中分别随机抽取样本
sampled_non_typical = random.sample(list(non_typical_indices), num_not_typical)
sampled_typical = random.sample(typical_indices, num_typical)
# 将抽取的样本索引合并
sampled_indices = sampled_non_typical + sampled_typical
random.shuffle(sampled_indices)
sampled_indices = np.array(sampled_indices).flatten()
# 获取 X_train 中这些采样样本的对应数据
X_sampled = X_train[sampled_indices]
y_sampled = y_train[sampled_indices]
remaining_indices = torch.tensor([i for i in range(5000) if i not in sampled_indices])
# 获取剩余的 2000 条数据
X_remaining = X_train[remaining_indices]
y_remaining = y_train[remaining_indices]
# 对剩余的 2000 条数据进行打乱
remaining_order = torch.randperm(len(X_remaining))  # 对剩余 2000 条数据进行打乱
X_remaining = X_remaining[remaining_order]  # 打乱后的剩余数据
y_remaining = y_remaining[remaining_order]  # 打乱后的剩余标签
# 获取 5000 条数据之后的数据（即原始数据中超过5000条的部分）
X_after_5000 = X_train[5000:]
y_after_5000 = y_train[5000:]
# 最终的 X_train 和 y_train：采样的 3000 条数据放前面，剩余数据打乱后跟上，最后是超过5000条的原始数据
X_train_reordered = torch.cat([X_sampled, X_remaining, X_after_5000])
y_train_reordered = torch.cat([y_sampled, y_remaining, y_after_5000])

Accum_True_RGN_sum = 0
acctime=0
for i in range(2000):
    start = time.time()
    if args.train_mode == 'ovr':
        # removal from all one-vs-rest models
        for k in range(y_train_onehot.size(1)):
            if weight is None or weight[i, k] > 0:
                X_rem = X_train_reordered[(i+1):]#剩余训练样本的特征矩阵
                y_rem = y_train_reordered[(i+1):, k]
                if weight is not None:
                    X_rem = X_rem[weight[(i+1):, k].gt(0)]
                    y_rem = y_rem[weight[(i+1):, k].gt(0)]
                #H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, args.lam) # 计算Hessian矩阵的逆
                H_inv = lr_gce_hessian_inv(w_approx[:, k], X_rem, y_rem, args.lam, q=args.q)
                #grad_i = lr_grad(w_approx[:, k], X_train_reordered[i].unsqueeze(0), y_train_reordered[i, k].unsqueeze(0), args.lam)# 计算梯度
                grad_i = lr_gce_grad(w_approx[:, k],
                                       X_train_reordered[i].unsqueeze(0),
                                       y_train_reordered[i, k].unsqueeze(0),
                                       args.lam,
                                       q=args.q)
                # 更新K矩阵，减去样本对K的贡献
                if weight is None:
                    K -= torch.ger(X_train_reordered[i], X_train_reordered[i])
                    spec_norm = spectral_norm(K)
                else:
                    Ks[k] -= torch.ger(X_train_reordered[i], X_train_reordered[i])
                    spec_norm = spectral_norm(Ks[k])
                Delta = H_inv.mv(grad_i)#Hessian 矩阵的逆矩阵与梯度的乘积，结果是权重的更新量Delta
                Delta_p = X_rem.mv(Delta)#更新量对剩余样本的整体影响
                w_approx[:, k] += Delta#更新第k类别的权重向量 w_approx，增加计算出的权重更新量 Delta
                grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm / 4).cpu()#梯度范数的近似计算
                Accum_True_RGN_sum = lr_gce_grad(w_approx[:, k], X_rem, y_rem, args.lam,q=args.q) + b[:, k]
    else:
        for k in range(y_train_onehot.size(1)):
            if weight is None or weight[i, k] > 0:
                X_rem = X_train[(i+1):]
                y_rem = y_train[(i+1):]
                H_inv = lr_hessian_inv(w_approx[:], X_rem, y_rem, args.lam)
                grad_i = lr_grad(w_approx, X_train[i].unsqueeze(0), y_train[i].unsqueeze(0), args.lam)
                if weight is None:
                    K -= torch.ger(X_train[i], X_train[i])
                    spec_norm = spectral_norm(K)
                else:
                    Ks[k] -= torch.ger(X_train[i], X_train[i])
                    spec_norm = spectral_norm(Ks[k])
                Delta = H_inv.mv(grad_i)
                Delta_p = X_rem.mv(Delta)
                w_approx += Delta
                grad_norm_approx[i] += (Delta.norm() * Delta_p.norm() * spec_norm / 4).cpu()

    Accum_True_RGN = Accum_True_RGN_sum.norm().item()
    cumulative_grad_norm = grad_norm_approx[:i + 1].sum().item()#累积梯度残差
    times[i] = time.time() - start
    acctime+=times[i]
    print('Iteration %d: Grad norm bound = %.6f, time = %.2fs,cumulative time= %.2fs,cumulative_grad_norm = %.6f,true gradient residual = %.6f' % (i + 1, grad_norm_approx[i], times[i],acctime, cumulative_grad_norm,Accum_True_RGN))

if args.train_mode == 'ovr':
    pred = X_test.mm(w_approx).max(1)[1]
    accuracy = pred.eq(y_test).float().mean().item()
    f1 = f1_score(y_test.cpu(), pred.cpu(), average='weighted')  # 计算加权平均的F1分数
    print('Test accuracy = %.4f' % accuracy)
    print('Test F1 score = %.4f' % f1)
else:
    pred = X_test.mv(w)
    accuracy = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    f1 = f1_score(y_test.cpu(), pred.gt(0).cpu(), average='binary')  # 计算二分类的F1分数
    print('Test accuracy = %.4f' % accuracy)
    print('Test F1 score = %.4f' % f1)

save_path = '%s/%s_%s_splits_%d_ratio_%.2f_std_%.1f_lam_%.0e_removal.pth' % (
    args.result_dir, args.extractor, args.dataset, args.train_splits, args.subsample_ratio, args.std, args.lam)
torch.save({'grad_norm_approx': grad_norm_approx, 'times': times}, save_path)
