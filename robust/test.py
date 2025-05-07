import torch

non_typical_samples = torch.load('non_typical_samples_svhn.pth')
typical_samples = torch.load('typical_samples_svhn.pth')

# 查看维度（即索引的个数）
print(non_typical_samples.size())
print(typical_samples.size())
print(non_typical_samples)
