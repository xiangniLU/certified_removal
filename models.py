# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
# from opacus.layers import DPLSTM
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Extractor(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, pool_size, bias=True, normalize=False):
        super(Extractor, self).__init__()
        self.normalize = normalize
        self.pool_size = pool_size
        conv_layers = []
        assert (len(num_channels) >= 2)
        self.conv_layers = nn.ModuleList([nn.Conv2d(num_channels[i], num_channels[i + 1],
                                                    kernel_size, stride, bias=bias) for i in
                                          range(len(num_channels) - 1)])

    def forward(self, x):
        for _, conv in enumerate(self.conv_layers):
            out = conv(x)
            x = F.max_pool2d(F.relu(out), self.pool_size, self.pool_size)
        out = x.view(x.size(0), -1)
        if self.normalize:
            out = F.normalize(out)
        return out


class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super(MLP, self).__init__()
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
        x = x.view(-1, self.input_size)
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        return self.output_layer(x)


class CNNFeatureExtractor(nn.Module):
    def __init__(self, channel_input=1, num_classes=2):
        super(CNNFeatureExtractor, self).__init__()
        # 输出通道数==卷积核个数, 卷积->BN->激活->池化
        # (n-k)/s+1
        self.conv1 = nn.Conv1d(in_channels=channel_input, out_channels=5, kernel_size=3,
                               stride=1)  # (1,102)--> (5,100)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=3, stride=1)  # (5,100)-->(5,98)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        out = x
        return out


class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_size=102, hidden_size=32):
        super(MLPFeatureExtractor, self).__init__()
        self.liner1 = nn.Linear(input_size, 64)
        self.liner2 = nn.Linear(64, hidden_size)
        self.Flatten = nn.Flatten()
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.Flatten(x)
        x = self.liner1(x)  # [bs,xxx],
        out = self.liner2(x)
        return out


# LSTM网络

# class LSTMFeatureExtractor(nn.Module):
#     def __init__(self, input_size=102, hidden_size=32, num_layers=2):
#         super(LSTMFeatureExtractor, self).__init__()
#         self.rnn = DPLSTM(input_size, hidden_size, num_layers, dropout=0.2, batch_first=True)
#
#     def forward(self, x):
#         r_out, (h_s, h_c) = self.rnn(x)
#         out = r_out[:, -1, :]
#
#         return out


class TfFeatureExtractor(nn.Module):
    def __init__(self, input_dim=102, hidden_dim=64, num_layers=1):
        super(TfFeatureExtractor, self).__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=input_dim, nhead=1, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        # self.liner = nn.Linear(input_dim, 5)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=1)  # 平均池化
        # x = self.liner (x)
        x = self.relu(x)
        return x
