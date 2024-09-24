# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPLSTM
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from fast_grad_conv import FastGradConv2d


class FastGradExtractor(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, pool_size, normalize=False):
        super(FastGradExtractor, self).__init__()
        self.normalize = normalize
        self.pool_size = pool_size
        conv_layers = []
        assert (len(num_channels) >= 2)
        self.conv_layers = nn.ModuleList([FastGradConv2d(num_channels[i], num_channels[i + 1],
                                                         kernel_size, stride) for i in range(len(num_channels) - 1)])

    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = F.max_pool2d(F.relu(conv(x)), self.pool_size, self.pool_size)
            # print(f"After conv layer {i}, shape: {x.shape}")  # 打印每个卷积层后的形状
        out = x.view(x.size(0), -1)
        if self.normalize:
            out = F.normalize(out)
        # print(f"Final output shape: {out.shape}")  # 打印最终输出的形状
        return out


# class MLPExtractor(nn.Module):
#     def __init__(self):
#         super(MLPExtractor, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(28*28, 128)
#         self.fc2 = nn.Linear(128, 64)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return x

class MLPExtractor(nn.Module):
    def __init__(self):
        super(MLPExtractor, self).__init__()
        self.flatten = nn.Flatten()
        self.liner1 = nn.Linear(3 * 32 * 32, 512) # svhn
        # self.liner1 = nn.Linear(3 * 128 * 128, 512) # lsun
        self.liner2 = nn.Linear(512, 128)
        # self.liner3 = nn.Linear(256, 64)
        self.Flatten = nn.Flatten()
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.Flatten(x)
        x = torch.relu(self.liner1(x))
        x = torch.relu(self.liner2(x))
        # x = torch.relu(self.liner3(x))
        out = x
        return out

# class LSTMExtractor(nn.Module):
#     def __init__(self, input_size=96, hidden_size=32, num_layers=2):
#         super(LSTMExtractor, self).__init__()
#         self.rnn = DPLSTM(input_size, hidden_size, num_layers, dropout=0.2, batch_first=True)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.permute(0, 2, 3, 1)  # 将形状变为 (batch_size, 32, 32, 3)
#         x = x.reshape(batch_size, 32, 32*3)  # 将形状变为 (batch_size, 32, 96)
#         r_out, (h_s, h_c) = self.rnn(x)
#         out = r_out[:, -1, :]  # 取最后一个时间步的输出
#         return out

class LSTMExtractor(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2):
        super(LSTMExtractor, self).__init__()
        self.rnn = DPLSTM(input_size, hidden_size, num_layers, dropout=0.2, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        batch_size = x.size(0)
        x = x.squeeze(1)  # 去除通道维度，变为 (batch_size, 28, 28)
        r_out, (h_s, h_c) = self.rnn(x)
        out = r_out[:, -1, :]  # 取最后一个时间步的输出
        return out

# class TfExtractor(nn.Module):
#     def __init__(self, input_dim=28, hidden_dim=64, num_layers=1):
#         super(TfExtractor, self).__init__()
#         self.encoder = TransformerEncoder(
#             TransformerEncoderLayer(d_model=input_dim, nhead=1, dim_feedforward=hidden_dim),
#             num_layers=num_layers
#         )
#         self.relu = nn.ReLU()
#         self._init_weights()
#
#     def _init_weights(self):
#         for param in self.encoder.parameters():
#             if param.dim() > 1:
#                 nn.init.xavier_uniform_(param)
#
#     def forward(self, x):
#         x = x.squeeze(1)  # 去掉通道维度 -> (batch_size, 28, 28)
#         x = self.encoder(x)
#         x = torch.mean(x, dim=1)  # 平均池化
#         x = self.relu(x)
#         return x

class TfExtractor(nn.Module):
    def __init__(self, input_dim=32*3, hidden_dim=64, num_layers=1):
        super(TfExtractor, self).__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=input_dim, nhead=1, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        # x: (batch_size, 3, 32, 32)
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # 变为 (batch_size, 32, 32, 3)
        x = x.reshape(batch_size, 32, 32*3)  # 变为 (batch_size, 32, 96)
        x = self.encoder(x)  # 变为 (batch_size, 32, 96)
        x = torch.mean(x, dim=1)  # 平均池化 -> (batch_size, 96)
        x = self.relu(x)
        return x


class FastGradMLP(nn.Module):
    """
    "Standard" MLP with support with goodfellow's backprop trick
    """

    def __init__(self, hidden_sizes):
        super(type(self), self).__init__()

        assert (len(hidden_sizes) >= 2)
        self.input_size = hidden_sizes[0]
        self.act = F.relu

        if len(hidden_sizes) == 2:
            self.hidden_layers = []
        else:
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 2)])
        self.output_layer = nn.Linear(hidden_sizes[-2], hidden_sizes[-1])

    def forward(self, x):
        out = x

        # Save the model inputs, which are considered the activations of the 0'th layer.
        activations = [out]
        linearCombs = []

        for layer in self.hidden_layers:
            linearComb = layer(out)
            out = self.act(linearComb)

            # Save the activations and linear combinations from this layer.
            activations.append(out)
            linearComb.requires_grad_(True)
            linearComb.retain_grad()
            linearCombs.append(linearComb)

        logits = self.output_layer(out)

        logits.requires_grad_(True)
        logits.retain_grad()
        linearCombs.append(logits)
        # print(logits)

        return (logits, activations, linearCombs)