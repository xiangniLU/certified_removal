import torch
from torchvision.datasets import MNIST, SVHN
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import os
import scipy.io
import numpy as np

'''
non_typical_indices = torch.load("../robust/non_typical_samples_mnist.pth")
typical_indices = torch.load("../robust/typical_samples_mnist.pth")
transform = transforms.ToTensor()
mnist = MNIST(root="../save", train=True, download=False, transform=transform)

# 取前10个最不典型样本（影响力最大）
top10_non_typical = non_typical_indices[-10:]

# 取后10个最典型样本（影响力最小）
bottom10_typical = typical_indices[:10]

output_dir = "FigResult/Mnist"
os.makedirs(output_dir, exist_ok=True)


def save_image_by_index(dataset, index_tensor, prefix):
    for i, idx in enumerate(index_tensor):
        img, label = dataset[idx.item()]         # 获取图像和标签
        img = img.squeeze(0).numpy()             # [1, 28, 28] → [28, 28]
        filename = os.path.join(output_dir, f"{prefix}_{i+1}.png")
        plt.imsave(filename, img, cmap='gray')
        print(f"Saved: {filename} (Label: {label})")

# === 5. 保存图像 === #
save_image_by_index(mnist, top10_non_typical, prefix="hard")    # 保存最不典型的前10张
save_image_by_index(mnist, bottom10_typical, prefix="normal")   # 保存最典型的后10张
'''

'''
df = pd.read_csv("../save/train.tsv", sep="\t", header=None, names=["sentence", "label"])

# 2. 加载索引
non_typical_indices = torch.load("../robust/non_typical_samples_sst.pth")
typical_indices = torch.load("../robust/typical_samples_sst.pth")

# 3. 取文本和标签
top10_non = df.iloc[non_typical_indices[-10:]]
bottom10_typ = df.iloc[typical_indices[:10]]

# 4. 输出并保存
output_dir = "FigResult/SST"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "top10_non_typical.txt"), "w", encoding="utf-8") as f:
    for i, row in enumerate(top10_non.itertuples(index=False)):
        f.write(f"[hard_{i+1}] {row.sentence} → sentence: {row.label}\n")

with open(os.path.join(output_dir, "bottom10_typical.txt"), "w", encoding="utf-8") as f:
    for i, row in enumerate(bottom10_typ.itertuples(index=False)):
        f.write(f"[normal_{i+1}] {row.sentence} → sentence: {row.label}\n")
'''

# 1. 加载 SVHN 图像数据
svhn_path = "../save/svhn/train_32x32.mat"
data = scipy.io.loadmat(svhn_path)
X = data['X']  # shape: (32, 32, 3, N)
y = data['y'].squeeze()  # shape: (N,)
y[y == 10] = 0  # 标签10表示数字0，改回来

# 2. 加载非典型和典型索引
non_typical_indices = torch.load("../robust/non_typical_samples_svhn.pth")
typical_indices = torch.load("../robust/typical_samples_svhn.pth")

# 3. 获取前10和后10
top10_non = non_typical_indices[-10:]
bottom10_typ = typical_indices[:10]

# 4. 设置输出路径
output_dir = "FigResult/SVHN"
os.makedirs(output_dir, exist_ok=True)

# 5. 保存图像函数
def save_svhn_images(indices, prefix):
    for i, idx in enumerate(indices):
        img = X[:, :, :, idx.item()]  # shape: (32, 32, 3)
        img = np.transpose(img, (1, 0, 2))  # 转置为 (H, W, C)
        label = y[idx.item()]
        filename = os.path.join(output_dir, f"{prefix}_{i+1}_label{label}.png")
        plt.imsave(filename, img.astype(np.uint8))
        print(f"Saved: {filename}")

# 6. 保存图像
save_svhn_images(top10_non, prefix="hard")
save_svhn_images(bottom10_typ, prefix="normal")