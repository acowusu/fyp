import torch

print("Hello World")
print("Hello World2")
from torch.cuda import is_available
print(is_available())
# print(torch.cuda.is_available())

# import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
np_array = np.array(data)
# x_np = torch.from_numpy(np.zeros((20000, 1000,20000)))
a = torch.zeros(1024, 3, 600, 600,3).to(device='cuda')
b = torch.zeros(512, 3, 600, 600,3).to(device='cuda')
c = torch.zeros(512, 3, 600, 600,3).to(device='cuda')

# b = torch.zeros(4096, 3, 600, 600,3).to(device='cuda')


print(a.shape)
print(b.shape)
print(c.shape)
