import torch
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

data = [[1,2],[3,4], [5,6]]

x_np = np.array(data)

x_data = torch.from_numpy(x_np)
x_gpu_data = torch.from_numpy(x_np).to(device)


x_data *= 2
x_gpu_data *= 5

x_ones = torch.ones_like(x_data)

x_rand = torch.rand_like(x_data, dtype=torch.float)



shape = (2,3,4,6)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor.dtype)
