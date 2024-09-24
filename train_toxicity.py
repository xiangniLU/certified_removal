from opacus.layers import DPLSTM
from sklearn.model_selection import train_test_split
import torch
from imblearn.over_sampling import SMOTE
import argparse
from collections import Counter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.optim as optim
# from attack import split_to_be_divisible, prepare_attack_data  # 导入自定义的模型
from fast_grad_models import FastGradExtractor, FastGradMLP  # 导入自定义的快速梯度模型
from train_func import train, train_private  # 导入训练函数
from test_func import test, test_linear  # 导入测试函数
from models import Extractor, MLP
import utils
import time
import os
from torchdp.privacy_analysis import compute_rdp, get_privacy_spent  # 导入隐私分析工具
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description='Training an toxcity model')
    # parser.add_argument('--data-dir', type=str, required=True, help='directory for toxcity data')  # SVHN 数据集的路径
    parser.add_argument('--save-dir', type=str, default='save', help='directory for saving trained model')  # 保存训练模型的路径
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')  # 训练时的批量大小
    parser.add_argument('--process-batch-size', type=int, default=50, help='batch size for processing')  # 处理时的批量大小
    parser.add_argument('--test-batch-size', type=int, default=50, help='batch size for testing')  # 测试时的批量大小
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')  # 训练的轮数
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # 学习率
    parser.add_argument('--lam', type=float, default=0, help='L2 regularization')  # L2 正则化项
    parser.add_argument('--std', type=float, default=5.0, help='noise multiplier for DP training')  # 差分隐私训练的噪声乘数
    parser.add_argument('--delta', type=float, default=1e-5, help='delta for DP training')  # 差分隐私训练的 delta
    parser.add_argument('--num-filters', type=int, default=10, help='number of conv filters')  # 卷积层的滤波器数量
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

    def parse_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        label_map = {'negative': 0, 'positive': 1}

        parsed_data = []
        for line in lines:
            parts = line.strip().split(';')
            features = list(map(int, parts[:-1]))  # 将特征转换为整数列表
            label = label_map[parts[-1]]  # 使用映射将标签转换为整数
            parsed_data.append(features + [label])  # 将标签添加到特征列表中

        return parsed_data

    data = parse_data('save/qsar_oral_toxicity.csv')

    X_train = [item[:-1] for item in data]
    y_train = [item[-1] for item in data]

    # 转换为PyTorch的张量
    smote = SMOTE(random_state=10)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


    # 转换为PyTorch的张量
    X_train_tensor = torch.tensor(X_resampled, dtype=torch.float32).unsqueeze(dim=1).to(device)
    y_train_tensor = torch.tensor(y_resampled, dtype=torch.long).view(-1, ).to(device)

    X_train, X_val, y_train, y_val = train_test_split(X_train_tensor, y_train_tensor, test_size=0.3, random_state=10)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(X_val, y_val)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(f"Batch {batch_idx} labels: {target}")

    class CNNFeatureExtractor(nn.Module):
        def __init__(self, channel_input=1, num_classes=2):
            super(CNNFeatureExtractor, self).__init__()
            # 输出通道数==卷积核个数, 卷积->BN->激活->池化
            # (n-k)/s+1
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 256, 128)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.relu = nn.ReLU(inplace=True)
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)  # (16,1024) --> (16,512)
            x = self.relu(self.conv2(x))
            x = self.pool(x)  # (32,512) --> (32,256)
            x = self.flatten(x)  # 展平
            x = self.relu(self.fc1(x))
            out = x
            return out

    class MLPFeatureExtractor(nn.Module):
        def __init__(self, input_size=1024, hidden_size=32):
            super(MLPFeatureExtractor, self).__init__()
            self.liner1 = nn.Linear(input_size, 256)
            self.liner2 = nn.Linear(256, hidden_size)
            self.Flatten = nn.Flatten()
            # self.sig = nn.Sigmoid()

        def forward(self, x):
            x = self.Flatten(x)
            x = self.liner1(x)  # [bs,xxx],
            out = self.liner2(x)
            return out

    class LSTMFeatureExtractor(nn.Module):
        def __init__(self, input_size=1024, hidden_size=128, num_layers=2):
            super(LSTMFeatureExtractor, self).__init__()
            self.rnn = DPLSTM(input_size, hidden_size, num_layers, dropout=0.2, batch_first=True)

        def forward(self, x):
            r_out, (h_s, h_c) = self.rnn(x)
            out = r_out[:, -1, :]

            return out

    class TfFeatureExtractor(nn.Module):
        def __init__(self, input_dim=1024, hidden_dim=64, num_layers=1):
            super(TfFeatureExtractor, self).__init__()
            self.encoder = TransformerEncoder(
                TransformerEncoderLayer(d_model=input_dim, nhead=1, dim_feedforward=hidden_dim),
                num_layers=num_layers
            )
            # self.liner = nn.Linear(input_dim, 5)
            self.relu = nn.ReLU()


        def forward(self, x):
            x = self.encoder(x)
            x = torch.mean(x, dim=1)  # 平均池化
            # x = self.liner (x)
            x = self.relu(x)
            return x
    # extr = CNNFeatureExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[128, 2]).to(device)

    # extr = MLPFeatureExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[32, 2]).to(device)

    # extr = LSTMFeatureExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[128, 2]).to(device)

    extr = TfFeatureExtractor().to(device)
    clf = FastGradMLP(hidden_sizes=[1024, 2]).to(device)

    # def print_model_parameters(model):
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"Layer: {name} | Shape: {param.shape} | Values: {param}")
    #
    # print_model_parameters(extr)

    loss_fn = lambda x, y: F.nll_loss(F.log_softmax(x, dim=1), y)

    # 设置保存路径
    # save_path = "/%s/ess_cnn_delta_%.2e_std_%.2f%s.pth" % (args.save_dir, args.delta, args.std, args.save_suffix)
    save_path = "save/result/toxcity_tf_DP_SGD.pth"

    # 如果模型文件不存在，则进行训练和保存
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
                       '%s/dp_delta_%.2e_std_%.2f_Toxicity_extracted.pth' % (
                       "save/result", args.delta, args.std))
        else:
            test(args, extr, clf, loss_fn, device, test_loader)



if __name__ == '__main__':
    main()



