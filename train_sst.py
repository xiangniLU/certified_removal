# 导入所需的库和模块
import nltk
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from gensim.models import Word2Vec
from nltk import word_tokenize
from models import Extractor, MLP, CNNFeatureExtractor, MLPFeatureExtractor, LSTMFeatureExtractor, \
    TfFeatureExtractor  # 导入自定义的模型
from fast_grad_models import FastGradExtractor, FastGradMLP  # 导入自定义的快速梯度模型
from train_func import train, train_private  # 导入训练函数
from test_func import test, test_linear  # 导入测试函数
import utils
import time
import os
from torchdp.privacy_analysis import compute_rdp, get_privacy_spent  # 导入隐私分析工具
from torch.utils.data import Dataset, DataLoader


def main():
    parser = argparse.ArgumentParser(description='Training an sst model')
    # parser.add_argument('--data-dir', type=str, required=True, help='directory for SVHN data')  # SVHN 数据集的路径
    parser.add_argument('--save-dir', type=str, default='save', help='directory for saving trained model')  # 保存训练模型的路径
    parser.add_argument('--batch-size', type=int, default=100, help='batch size for training')  # 训练时的批量大小
    parser.add_argument('--process-batch-size', type=int, default=50, help='batch size for processing')  # 处理时的批量大小
    parser.add_argument('--test-batch-size', type=int, default=100, help='batch size for testing')  # 测试时的批量大小
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')  # 训练的轮数
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')  # 学习率
    parser.add_argument('--lam', type=float, default=0, help='L2 regularization')  # L2 正则化项
    parser.add_argument('--std', type=float, default=6.0, help='noise multiplier for DP training')  # 差分隐私训练的噪声乘数
    parser.add_argument('--delta', type=float, default=1e-5, help='delta for DP training')  # 差分隐私训练的 delta
    parser.add_argument('--num-filters', type=int, default=5, help='number of conv filters')  # 卷积层的滤波器数量
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

    nltk.download('punkt')
    # 设置 DataLoader 使用的参数
    kwargs = {'num_workers': 0, 'pin_memory': True}

    # 读取和预处理数据
    class SSTDataset(Dataset):
        def __init__(self, file_path):
            self.data = []
            self.targets = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    label, text = line.strip().split('\t')
                    self.data.append(text)
                    self.targets.append(int(label))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data[idx]
            label = self.targets[idx]
            return text, label

    # 训练Word2Vec模型
    def train_word2vec(sentences, vector_size=102, window=5, min_count=1, workers=4):
        model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        return model

    # 句子向量化
    def vectorize_sentence(sentence, model, vector_size):
        words = word_tokenize(sentence)
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if len(word_vectors) == 0:  # 如果句子中没有单词在词汇表中，返回零向量
            return np.zeros(vector_size)
        sentence_vector = np.mean(word_vectors, axis=0)
        return sentence_vector

    # 加载数据并进行预处理
    train_file = 'save/train.tsv'
    test_file = 'save/test.tsv'

    train_dataset = SSTDataset(train_file)
    test_dataset = SSTDataset(test_file)

    # 获取所有的句子
    all_sentences = [word_tokenize(text) for text, _ in train_dataset] + [word_tokenize(text) for text, _ in
                                                                          test_dataset]
    # 训练Word2Vec模型
    vector_size = 102
    word2vec_model = train_word2vec(all_sentences, vector_size=vector_size)

    # 向量化数据集
    class VectorizedSSTDataset(Dataset):
        def __init__(self, original_dataset, word2vec_model, vector_size, device):
            self.data = []
            self.targets = original_dataset.targets
            self.vector_size = vector_size
            self.device = device
            for text, _ in original_dataset:
                vectorized_text = vectorize_sentence(text, word2vec_model, vector_size)
                self.data.append(vectorized_text)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            data_tensor = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0).to(
                self.device)  # 添加额外的维度并移动到device
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.long).to(self.device)
            return data_tensor, target_tensor  # 返回元组

    # 向量化训练和测试数据集
    vectorized_train_dataset = VectorizedSSTDataset(train_dataset, word2vec_model, vector_size, device)
    vectorized_test_dataset = VectorizedSSTDataset(test_dataset, word2vec_model, vector_size, device)

    # 创建数据加载器
    train_loader = DataLoader(vectorized_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(vectorized_test_dataset, batch_size=args.batch_size, shuffle=False)

    # extr = CNNFeatureExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[980, 2]).to(device)

    # extr = MLPFeatureExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[32, 2]).to(device)

    extr = LSTMFeatureExtractor().to(device)
    clf = FastGradMLP(hidden_sizes=[32, 2]).to(device)

    # extr = TfFeatureExtractor().to(device)
    # clf = FastGradMLP(hidden_sizes=[102, 2]).to(device)

    # 损失函数
    loss_fn = lambda x, y: F.nll_loss(F.log_softmax(x, dim=1), y)

    # 设置保存路径
    # save_path = "/content/drive/MyDrive/certified-removal-main/%s/sst_cnn_delta_%.2e_std_%.2f%s.pth" % (args.save_dir, args.delta, args.std, args.save_suffix)
    save_path = "save/result/sst_lstm_DP_SGD.pth"

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
                       '%s/dp_delta_%.2e_std_%.2f_SST_extracted.pth' % ("save/result", args.delta, args.std))
            # torch.save({'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test},
            #            '%s/dp_delta_%.2e_std_%.2f_SST_extracted.pth' % (
            #            "/content/drive/MyDrive/certified-removal-main/save", args.delta, args.std))
        else:
            test(args, extr, clf, loss_fn, device, test_loader)


if __name__ == '__main__':
    main()
