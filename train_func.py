# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
import math
from utils import per_example_gradient, clip_and_sum_gradients, add_noisy_gradient, batch_grads_to_vec, params_to_vec, vec_to_params, compute_full_grad, loss_with_reg


# trains a regular model for a single epoch
# sst_cnn
def train(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch, verbose=True):
    # 将额外特征提取器（如果存在）和分类器设置为训练模式
    if extr is not None:
        extr.train()
    clf.train()

    # 初始化变量以跟踪总损失和正确预测的数量
    total_loss = 0
    correct = 0

    # 遍历训练数据加载器中的每个批次
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 将优化器的梯度缓存清零
        optimizer.zero_grad()
        # 如果存在-额外特征提取器，使用提取器对输入数据进行特征提取，并将结果传递给分类器
        if extr is not None:
            output = clf(extr(data))
            # 如果分类器的输出长度为3，则选择第一个输出
            if len(output) == 3:
                output = output[0]
                # print(output[0])
        else:
            output = clf(data)
        loss = loss_fn(output, target)
        # 如果 args.lam > 0，则添加 L2 正则化损失
        if args.lam > 0:
            if extr is not None:
                loss += args.lam * params_to_vec(extr.parameters()).pow(2).sum() / 2
            loss += args.lam * params_to_vec(clf.parameters()).pow(2).sum() / 2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 计算预测结果
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        # 如果 verbose 为 True 并且达到记录间隔，则打印训练进度和损失
        if verbose and (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}%'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * len(data))))

'''
def train(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch, verbose=True):
    # 将额外特征提取器（如果存在）和分类器设置为训练模式
    if extr is not None:
        extr.train()
    clf.train()
    # 遍历训练数据加载器中的每个批次
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, target = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        # 将优化器的梯度缓存清零
        optimizer.zero_grad()
        # 如果存在额外特征提取器，使用提取器对输入数据进行特征提取，并将结果传递给分类器
        if extr is not None:
            output = clf(extr(input_ids=input_ids, attention_mask=attention_mask))
            # 如果分类器的输出长度为3，则选择第一个输出
            if len(output) == 3:
                output = output[0]
        else:
            output = clf(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, target)
        if args.lam > 0:
            if extr is not None:
                loss += args.lam * params_to_vec(extr.parameters()).pow(2).sum() / 2
            loss += args.lam * params_to_vec(clf.parameters()).pow(2).sum() / 2
        loss.backward()
        optimizer.step()
        if verbose and (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(input_ids), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
'''
# trains a private model for a single epoch using private SGD
# clf must be a FastGradMLP
'''
def train_private(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch, C, std, include_linear=False, verbose=True):
    extr.train()  # 设置提取器模型为训练模式
    clf.train()  # 设置分类器模型为训练模式
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, target = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)  # 将数据和标签移动到训练设备上
        optimizer.zero_grad()  # 梯度清零
        # 计算每个样本的梯度
        num_batches = int(math.ceil(float(input_ids.size(0)) / args.process_batch_size))
        loss = 0
        grad_vec = None
        for i in range(num_batches):
            start = i * args.process_batch_size
            end = min((i + 1) * args.process_batch_size, input_ids.size(0))
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            target_batch = target[start:end]
            # 计算每个样本的损失和梯度
            loss_batch, gradients_batch = per_example_gradient(extr, clf, input_ids_batch, attention_mask_batch, target_batch, loss_fn,
                                                               include_linear=include_linear)
            loss += input_ids_batch.size(0) * loss_batch.item()  # 累积损失
            if i == 0:
                grad_vec = clip_and_sum_gradients(gradients_batch, C)  # 裁剪并累积梯度
            else:
                grad_vec += clip_and_sum_gradients(gradients_batch, C)  # 裁剪并累积梯度
        loss /= input_ids.size(0)  # 计算平均损失
        grad_vec /= input_ids.size(0)  # 计算平均梯度
        noise = add_noisy_gradient(extr, clf, device, grad_vec, C, std / input_ids.size(0),
                                   include_linear=include_linear)  # 添加噪声
        optimizer.step()  # 更新模型参数
        if verbose and (batch_idx + 1) % args.log_interval == 0:  # 输出训练信息
            print('Epoch %d [%d/%d]: loss = %.4f, grad_norm = %.4f, noise_norm = %.4f' % (
                epoch, (batch_idx + 1) * len(input_ids), len(train_loader.dataset), loss,
                grad_vec.norm(), noise.norm()))
'''


def train_private(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch, C, std, include_linear=False,
                  verbose=True):
    extr.train()
    clf.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        num_batches = int(math.ceil(float(data.size(0)) / args.process_batch_size))
        loss = 0
        grad_vec = None
        for i in range(num_batches):
            start = i * args.process_batch_size
            end = min((i + 1) * args.process_batch_size, data.size(0))
            data_batch = data[start:end]
            target_batch = target[start:end]
            loss_batch, gradients_batch = per_example_gradient(extr, clf, data_batch, target_batch, loss_fn,
                                                               include_linear=include_linear)
            # print(f"Batch {i+1}/{num_batches}, Gradients batch length: {len(gradients_batch)}")  # 添加调试信息
            # if len(gradients_batch) == 0:
            #     print("No gradients found in this batch!")  # 添加调试信息
            loss += data_batch.size(0) * loss_batch.item()
            if i == 0:
                grad_vec = clip_and_sum_gradients(gradients_batch, C)
            else:
                grad_vec += clip_and_sum_gradients(gradients_batch, C)

        loss /= data.size(0)
        grad_vec /= data.size(0)
        noise = add_noisy_gradient(extr, clf, device, grad_vec, C, std / data.size(0), include_linear=include_linear)
        optimizer.step()

        # 计算当前 batch 的预测准确率
        with torch.no_grad():
            logits, _, _ = clf(extr(data))  # 获取 logits
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            accuracy = 100. * correct / total

        if verbose and (batch_idx + 1) % args.log_interval == 0:
            print('Epoch %d [%d/%d]: loss = %.4f, grad_norm = %.4f, noise_norm = %.4f, accuracy = %.2f%%' % (
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), loss,
                grad_vec.norm(), noise.norm(), accuracy))

