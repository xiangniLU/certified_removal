import os
import random
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
'''
# 定义文件夹路径
classroom_dir = 'save/LSUN/classroom'
conference_room_dir = 'save/LSUN/conference_room'
output_dir = 'save/lsun_'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取文件夹中的图片文件
classroom_images = [os.path.join(classroom_dir, f) for f in os.listdir(classroom_dir) if f.endswith('.webp')]
conference_room_images = [os.path.join(conference_room_dir, f) for f in os.listdir(conference_room_dir) if
                          f.endswith('.webp')]

# 确保conference_room中有足够的图片
if len(conference_room_images) < len(classroom_images):
    raise ValueError("conference_room文件夹中的图片数量不足")

# 抽取相同数量的图片
conference_room_images = random.sample(conference_room_images, len(classroom_images))

# 添加标签 (0: classroom, 1: conference_room)
classroom_labels = [(img, 0) for img in classroom_images]
conference_room_labels = [(img, 1) for img in conference_room_images]

# 合并图片和标签列表
combined_images_labels = classroom_labels + conference_room_labels

# 打乱图片和标签顺序
random.shuffle(combined_images_labels)

# 提取图片和标签
combined_images, combined_labels = zip(*combined_images_labels)

# 划分数据集，80%用于训练，20%用于测试
train_images, test_images, train_labels, test_labels = train_test_split(combined_images, combined_labels, test_size=0.2,
                                                                        random_state=42)

# 创建训练集和测试集目录
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# 保存图片并创建标签文件
def save_images_and_labels(images, labels, output_path):
    os.makedirs(output_path, exist_ok=True)
    label_file_path = os.path.join(output_path, 'labels.txt')

    with open(label_file_path, 'w') as label_file:
        for img, label in zip(images, labels):
            # 复制图片到输出目录
            img_name = os.path.basename(img)
            shutil.copy(img, os.path.join(output_path, img_name))

            # 写入标签文件
            label_file.write(f'{img_name} {label}\n')


# 保存训练集图片和标签
save_images_and_labels(train_images, train_labels, train_dir)

# 保存测试集图片和标签
save_images_and_labels(test_images, test_labels, test_dir)

print(f'训练集和测试集已生成并保存在 {output_dir} 目录中')
'''
class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        with open(label_file, 'r') as f:
            for line in f:
                image_name, label = line.strip().split()
                self.image_paths.append(os.path.join(image_dir, image_name))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

import torch

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小为 224x224
    transforms.ToTensor(),  # 转换为 PyTorch 张量
])

# 创建训练集和测试集的数据集对象
train_dataset = CustomDataset(image_dir='save/lsun_/train', label_file='save/lsun_/train/labels.txt', transform=transform)
test_dataset = CustomDataset(image_dir='save/lsun_/test', label_file='save/lsun_/test/labels.txt', transform=transform)
