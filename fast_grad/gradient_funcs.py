# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
import pdb  # debugging

from goodfellow_backprop import goodfellow_backprop


def full(model, X, y):
    """
    计算完整目标函数的梯度
    """

    logits, _, _ = model.forward(X)
    loss = F.binary_cross_entropy_with_logits(logits.view((-1,)), y)
    grad = torch.autograd.grad(loss, model.parameters())

    return grad


def naive(model, X, y):
    """
	以完整批次方式计算预测值，
    然后在各个损失上调用 backward
	"""

    grad_list = []
    logits, _, _ = model.forward(X)
    N = X.shape[0]
    for n in range(N):
        model.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits[n], y[n].view(-1, ))
        loss.backward(retain_graph=True)

        grad_list.append(list([p.grad.clone() for p in model.parameters()]))

    grads = []
    for p_id in range(len(list(model.parameters()))):
        grads.append(torch.cat([grad_list[n][p_id].unsqueeze(0) for n in range(N)]))

    return grads


def goodfellow(model, X, y):
    """
	使用 Goodfellow 的技巧计算单个梯度。
    参考: Efficient per-example gradient computations
    链接: https://arxiv.org/abs/1510.01799
	"""
    model.zero_grad()

    logits, activations, linearCombs = model.forward(X)
    loss = F.binary_cross_entropy_with_logits(logits.view((-1,)), y)

    linearGrads = torch.autograd.grad(loss, linearCombs)
    gradients = goodfellow_backprop(activations, linearGrads)

    return gradients
