import torch
import numpy as np
from torch.autograd import Variable

# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# torch2numpy = torch_data.numpy()
#
# print(
#     "\nnp_data", np_data,
#     "\ntorch_data", torch_data,
#     "\ntorch2numpy", torch2numpy
# )

# # abs  绝对值
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)
# print(
#     "\nabs",
#     "\nnumpy", np.mean(data),
#     "\ntorch", torch.mean(tensor)
# )


# data = [[1, 2], [3, 4]]
# tensor = torch.FloatTensor(data)
# print(
#     "\nnumpy", np.matmul(data, data),          # matmul 两矩阵相乘
#     "\ntorch", torch.mm(tensor, tensor),            # mm
#
# )


tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad = True)
t_out = torch.mean(tensor*tensor)    # x^2
v_out = torch.mean(variable*variable)
print(t_out)
print(v_out)
# v_out.backward()
# print(variable.grad)
print(variable)
print(variable.data)
print(variable.data.numpy())
