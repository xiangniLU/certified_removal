# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pdb
import helpers
from gradient_funcs import full, goodfellow, naive

# 定义运行函数 runWith，接受样本数 N、输入和隐藏层维度 D、隐藏层数量 L 作为参数
def runWith(N, D, L):
	# 生成数据和模型，返回样本数据 X、标签 y 和模型 model
	X, y, model = helpers.make_data_and_model(N, D, L)

	# 定义梯度计算方法的名称和对应的函数
	names = ["Goodf", "Naive"]
	methods = [goodfellow, naive]

	# 检查不同梯度计算方法的正确性
	helpers.check_correctness(full, names, methods, model, X, y)

	# 进行简单的计时
	helpers.simpleTiming(full, names, methods, model, X, y, REPEATS=1)

	# 若要进行性能分析，可以取消下一行的注释
	#helpers.profiling(full, names, methods, model, X, y)

# 不同参数设置的列表，每个设置包含 N、D、L
setups = [
	[2,3,1],
	[10,100,10],
	[100,100,10],
	[100,300,3],
	[32,300,50],
	[1000,100,10]
]

print("README:")
print()
print("Parameters:")
print("- N: Number of samples")  # 样本数
print("- D: Dimensionality of the inputs and hidden layers - width of the network")  # 输入和隐藏层维度 - 网络的宽度
print("- L: Number of hidden layers - depth of the network")  # 隐藏层数量 - 网络的深度
print()
print("Functions:")
print("- Full : Computes the averaged gradient")  # 计算平均梯度
print("- Naive: Compute each individual gradient by repeatedly calling backward")  # 通过重复调用 backward 计算每个单独的梯度
print("- Goodf: Compute the individual gradients using Goodfellow's Trick,")  # 使用 Goodfellow 的技巧计算单个梯度
print("  which is equivalent to redefining the backward pass to _not_ aggregate individual gradients")  # 该技巧等效于重新定义 backward pass，_不_聚合单个梯度
print()
print("Checking correctness is done with torch.norm()")
print("- For the diff. to the Full gradient, we first average over the sample")  # 对于与 Full 梯度的差异，首先对样本进行平均
print("- For the difference between individual gradient methods,")  # 对于各个梯度方法之间的差异
print("  we take the L2 norm between [N x ...] matrices")  # 我们计算 [N x ...] 矩阵之间的 L2 范数

# 遍历不同参数设置，调用 runWith 函数运行测试
for setup in setups:
	print()
	print("Setup [N, D, L] =", setup)
	print("---")
	runWith(*setup)
