import torch
import numpy as np

print("start")

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

tensor = torch.ones(4,4).to(device)

tensor[:,2] = 9
tensor[1] *= 2

catted_tensor = torch.cat((tensor, tensor), dim=1)

mult_tensor = tensor @ catted_tensor @ catted_tensor.T


tensor_to_sum = torch.tensor([[1,2,3],[4,5,6]])

tensor_to_sum.sub_(2)

mnist_short_tensor = torch.rand([4, 1, 6, 6])

new_tensor = torch.zeros(4,3)

inx = torch.tensor([[2], [1], [1], [0]])

new_tensor.scatter_(1, inx, 2)

target = torch.empty(3, dtype=torch.long).random_(5)

print(target)