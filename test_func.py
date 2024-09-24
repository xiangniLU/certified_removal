# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from train_func import train
from utils import extract_features
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


class Linear(nn.Module):
    def __init__(self, input_size):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_size, 10)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
'''
def test(args, extr, clf, loss_fn, device, test_loader, verbose=True):
    if extr is not None:
        extr.eval()
    clf.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, target = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            if extr is not None:
                # print("extr is not None!")
                output = clf(extr(input_ids=input_ids, attention_mask=attention_mask))
                # print("output:",output)
                if len(output) == 3:
                    # print("len(output) == 3")
                    output = output[0]
                    # print("output:",output[0])
            else:
                # print("extr is None")
                output = clf(input_ids=input_ids, attention_mask=attention_mask)
            test_loss += output.size(0) * loss_fn(output, target).item()
            if output.size(1) > 1:
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            else:
                pred = output.gt(0).long()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = float(correct) / len(test_loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100 * test_acc))
    return test_loss, test_acc
'''
# computes test accuracy
def test(args, extr, clf, loss_fn, device, test_loader, verbose=True):
    if extr is not None:
        extr.eval()
    clf.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if extr is not None:
                output = clf(extr(data))
                if len(output) == 3:
                    output = output[0]
            else:
                output = clf(data)
            test_loss += output.size(0) * loss_fn(output, target).item()
            if output.size(1) > 1:
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            else:
                pred = output.gt(0).long()
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = float(correct) / len(test_loader.dataset)

    # Calculate F1-score
    f1 = f1_score(all_targets, all_preds, average='weighted')

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1-score: {:.4f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100 * test_acc, f1))
    return test_loss, test_acc, f1
    # 它接收一个已经训练好的特征提取器（extr）和一个分类器（clf）作为输入。
    # 对于测试集中的每个样本，它首先使用特征提取器对输入数据进行特征提取，然后使用分类器对提取的特征进行预测。
    # 它计算并返回测试损失和准确率。

# computes test accuracy of a logistic regression model using features extracted from extr
def test_linear(args, extr, device, train_loader, test_loader, verbose=True):
    # 使用提取器对训练集进行特征提取
    X_train, y_train = extract_features(extr, device, train_loader)
    # 使用提取器对测试集进行特征提取
    X_test, y_test = extract_features(extr, device, test_loader)
    # 初始化逻辑回归模型，设置正则化参数 C
    clf = LogisticRegression(C=1 / (X_train.size(0) * args.lam), solver='saga', multi_class='multinomial',
                             verbose=int(verbose))
    # 在提取的特征上训练逻辑回归模型
    clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    # 在测试集上评估模型性能，并计算准确率
    acc = clf.score(X_test.cpu().numpy(), y_test.cpu().numpy())
    # 打印测试准确率
    print('Test accuracy = %.4f' % acc)
    return acc  # 返回测试准确率
'''
def test_linear(args, extr, device, train_loader, test_loader, verbose=True):
    X_train, y_train = extract_features(extr, device, train_loader)
    X_test, y_test = extract_features(extr, device, test_loader)
    clf = LogisticRegression(C=1/(X_train.size(0)*args.lam), solver='saga', multi_class='multinomial', verbose=int(verbose))
    clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    acc = clf.score(X_test.cpu().numpy(), y_test.cpu().numpy())
    print('Test accuracy = %.4f' % acc)
    return acc
    # 首先，它使用提取器对训练集和测试集进行特征提取(使用本地数据)，从而将原始数据转换为特征向量。
    # 然后，它初始化一个逻辑回归模型(使用本地数据)，并在提取的特征上进行训练。
    # 最后，它在测试集上评估模型的性能，并返回测试准确率
'''
