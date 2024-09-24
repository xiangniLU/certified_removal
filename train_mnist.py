# 导入所需的库和模块
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import argparse
from models import Extractor, MLP  # 导入自定义的模型
from fast_grad_models import FastGradExtractor, FastGradMLP, MLPExtractor, LSTMExtractor, TfExtractor  # 导入自定义的快速梯度模型
from train_func import train, train_private  # 导入训练函数
from test_func import test, test_linear  # 导入测试函数
import utils
import time
import os
from torchdp.privacy_analysis import compute_rdp, get_privacy_spent  # 导入隐私分析工具


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Training an Mnist model')
    parser.add_argument('--data-dir', type=str, required=True, help='directory for Mnist data')  # mnist 数据集的路径
    parser.add_argument('--save-dir', type=str, default='save', help='directory for saving trained model')  # 保存训练模型的路径
    parser.add_argument('--batch-size', type=int, default=100, help='batch size for training')  # 训练时的批量大小
    parser.add_argument('--process-batch-size', type=int, default=100, help='batch size for processing')  # 处理时的批量大小
    parser.add_argument('--test-batch-size', type=int, default=200, help='batch size for testing')  # 测试时的批量大小
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')  # 训练的轮数
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # 学习率  cnn0.001  lstm 0.005
    parser.add_argument('--lam', type=float, default=0, help='L2 regularization')  # L2 正则化项
    parser.add_argument('--std', type=float, default=6.0, help='noise multiplier for DP training')  # 差分隐私训练的噪声乘数
    parser.add_argument('--delta', type=float, default=1e-5, help='delta for DP training')  # 差分隐私训练的 delta
    parser.add_argument('--num-filters', type=int, default=32, help='number of conv filters')  # 卷积层的滤波器数量
    parser.add_argument('--seed', type=int, default=1, help='manual random seed')  # 随机种子
    parser.add_argument('--log-interval', type=int, default=10, help='logging interval')  # 日志输出的间隔
    parser.add_argument('--train-mode', type=str, default='default',
                        help='train mode [default/private/full_private]')  # 训练模式
    parser.add_argument('--test-mode', type=str, default='default', help='test mode [default/linear/extract]')  # 测试模式
    parser.add_argument('--save-suffix', type=str, default='', help='suffix for model name')  # 模型名称的后缀
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='normalize extracted features')  # 是否对提取的特征进行归一化
    parser.add_argument('--single-layer', action='store_true', default=False,
                        help='single convolutional layer')  # 是否使用单卷积层
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='for saving the trained model')  # 是否保存训练得到的模型
    args = parser.parse_args()

    # 设置随机种子和设备
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # 设置 DataLoader 使用的参数
    kwargs = {'num_workers': 0, 'pin_memory': True}

    # 数据预处理
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载训练集、额外集和测试集
    trainset = torchvision.datasets.MNIST(args.data_dir, train=True, download=True, transform=mnist_transform)
    testset = torchvision.datasets.MNIST(args.data_dir, train=False, download=True, transform=mnist_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # CNN
    if args.single_layer:
        extr = FastGradExtractor([1, args.num_filters], 9, 1, 2, normalize=args.normalize).to(device)
        clf = FastGradMLP([12 * 12 * args.num_filters, 10]).to(device)
    else:
        extr = FastGradExtractor([1, args.num_filters, args.num_filters], 5, 1, 2, normalize=args.normalize).to(device)
        clf = FastGradMLP([4 * 4 * args.num_filters, 10]).to(device)

    #  MLP
    # extr = MLPExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[64, 10]).to(device)

    # lstm
    # extr = LSTMExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[128, 10]).to(device)

    # transfromer
    # extr = TfExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[28,10]).to(device)

    # 损失函数
    loss_fn = lambda x, y: F.nll_loss(F.log_softmax(x, dim=1), y)

    # 设置保存路径
    # save_path = "/content/drive/MyDrive/certified-removal-main/%s/svhn_cnn_delta_%.2e_std_%.2f%s.pth" % (args.save_dir, args.delta, args.std, args.save_suffix)
    save_path = "save/result/mnist_cnn_DP_SGD.pth"


    if not os.path.exists(save_path):
        # 优化器
        optimizer = optim.Adam(list(extr.parameters()) + list(clf.parameters()), lr=args.lr, weight_decay=args.lam)

        # 隐私参数设置
        C = 4
        n = len(train_loader.dataset)
        q = float(args.batch_size) / float(n)
        T = args.epochs * len(train_loader)

        # 使用 RDP 分析计算隐私损失
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                  list(range(5, 64)) + [128, 256, 512, 1024, 2048, 4096])
        epsilon, _ = get_privacy_spent(orders, compute_rdp(q, args.std, T, orders), args.delta)
        print('RDP computed privacy loss: epsilon = %.2f at delta = %.2e' % (epsilon, args.delta))

        # 训练模型
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            if args.train_mode == 'private' or args.train_mode == 'full_private':
                include_linear = (args.train_mode == 'full_private')
                train_private(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch, C, args.std,
                              include_linear=include_linear)
            else:
                train(args, extr, clf, loss_fn, device, train_loader, optimizer, epoch)
            test(args, extr, clf, loss_fn, device, test_loader)
        print(time.time() - start)

        # 保存模型
        if args.save_model:
            torch.save({'extr': extr.state_dict(), 'clf': clf.state_dict()}, save_path)

    # 如果模型文件存在，则加载模型并进行测试
    else:
        checkpoint = torch.load(save_path)
        extr.load_state_dict(checkpoint['extr'])
        clf.load_state_dict(checkpoint['clf'])
        if args.test_mode == 'linear':
            test_linear(args, extr, device, train_loader, test_loader)
        elif args.test_mode == 'extract':
            # 提取特征以供训练启用移除的线性模型
            X_train, y_train = utils.extract_features(extr, device, train_loader)
            X_test, y_test = utils.extract_features(extr, device, test_loader)
            torch.save({'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test},
                       '%s/dp_delta_%.2e_std_%.2f_Mnist_extracted.pth' % ("save/result", args.delta, args.std))
        else:
            test(args, extr, clf, loss_fn, device, test_loader)


if __name__ == '__main__':
    main()
