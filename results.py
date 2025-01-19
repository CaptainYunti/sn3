import torch
from torch.utils.data import DataLoader
from torch import nn
from load_data import test_data
from train_test import test
import my_models


test_dataloader = DataLoader(test_data)
loss_fn = nn.CrossEntropyLoss()


#####################################################
# Liczba neuronow #
#####################################################

model_128 = my_models.OneHiddenLayer(128)
model_128.load_state_dict(torch.load("models/model_128_relu_sgd_10.pth", weights_only=True))

model_256 = my_models.OneHiddenLayer(256)
model_256.load_state_dict(torch.load("models/model_256_relu_sgd_10.pth", weights_only=True))

model_512 = my_models.OneHiddenLayer(512)
model_512.load_state_dict(torch.load("models/model_512_relu_sgd_10.pth", weights_only=True))

print("\nPorownanie liczby neuronow (1 warstwa, 10 epochs): ")
print("128:")
test(test_dataloader, model_128, loss_fn)
print("256:")
test(test_dataloader, model_256, loss_fn)
print("512:")
test(test_dataloader, model_512, loss_fn)




model_512_256 = my_models.TwoHiddenLayer((512, 256))
model_512_256.load_state_dict(torch.load("models/model_512_256_relu_sgd_10.pth", weights_only=True))

model_512_512 = my_models.TwoHiddenLayer((512, 512))
model_512_512.load_state_dict(torch.load("models/model_512_512_relu_sgd_10.pth", weights_only=True))

print("\n\nPorownanie liczby neuronow (2 warstwy, 10 epochs): ")
print("512, 256:")
test(test_dataloader, model_512_256, loss_fn)
print("512, 512:")
test(test_dataloader, model_512_512, loss_fn)



model_512_512_512 = my_models.ThreeHiddenLayer((512,512,512))
model_512_512_512.load_state_dict(torch.load("models/model_512_512_512_relu_sgd_10.pth", weights_only=True))

model_512_512_128 = my_models.ThreeHiddenLayer((512,512,128))
model_512_512_128.load_state_dict(torch.load("models/model_512_512_128_relu_sgd_10.pth", weights_only=True))

print("\n\nPorownanie liczby neuronow (3 warstwy, 10 epochs): ")
print("512, 512, 128:")
test(test_dataloader, model_512_512_128, loss_fn)
print("512, 512, 512:")
test(test_dataloader, model_512_512_512, loss_fn)


#####################################################
# Liczba warstw #
#####################################################


model_512.load_state_dict(torch.load("models/model_512_relu_sgd_30.pth", weights_only=True))
model_512_512.load_state_dict(torch.load("models/model_512_512_relu_sgd_30.pth", weights_only=True))
model_512_512_128.load_state_dict(torch.load("models/model_512_512_128_relu_sgd_30.pth", weights_only=True))

print("\n\nPorownanie liczby warstw (30 epochs): ")
print("1 warstwa (512):")
test(test_dataloader, model_512, loss_fn)
print("2 warstwy (512, 512):")
test(test_dataloader, model_512_512, loss_fn)
print("3 warstwy (512, 512, 128):")
test(test_dataloader, model_512_512_128, loss_fn)


#####################################################
# Optimizer #
#####################################################


model_512_sgd = my_models.OneHiddenLayer(512)
model_512_sgd.load_state_dict(torch.load("models/model_512_relu_sgd_40.pth", weights_only=True))

model_512_adadelta = my_models.OneHiddenLayer(512)
model_512_adadelta.load_state_dict(torch.load("models/model_512_relu_adadelta_40.pth", weights_only=True))

model_512_adam = my_models.OneHiddenLayer(512)
model_512_adam.load_state_dict(torch.load("models/model_512_relu_adam_40.pth", weights_only=True))

print("\n\nPorownanie opimizera (40 epochs): ")
print("SGD (512):")
test(test_dataloader, model_512_sgd, loss_fn)
print("AdaDelta (512):")
test(test_dataloader, model_512_adadelta, loss_fn)
print("Adam (512)):")
test(test_dataloader, model_512_adam, loss_fn)


#####################################################
# Funkcja Aktywacji #
#####################################################

model_relu = my_models.OneHiddenLayer(512)
model_relu.load_state_dict(torch.load("models/model_512_relu_adam_16.pth", weights_only=True))

model_sigmoid = my_models.OneHiddenLayer(512, nn.Sigmoid)
model_sigmoid.load_state_dict(torch.load("models/model_512_sigmoid_adam_16.pth", weights_only=True))

model_tanh = my_models.OneHiddenLayer(512, nn.Tanh)
model_tanh.load_state_dict(torch.load("models/model_512_tanh_adam_16.pth", weights_only=True))

model_elu = my_models.OneHiddenLayer(512, nn.ELU)
model_elu.load_state_dict(torch.load("models/model_512_elu_adam_16.pth", weights_only=True))



print("\n\nPorownanie funkcji aktywacji (16epochs, optimizer - Adam): ")
print("ReLU:")
test(test_dataloader, model_relu, loss_fn)
print("Sigmoid:")
test(test_dataloader, model_sigmoid, loss_fn)
print("Tanh:")
test(test_dataloader, model_tanh, loss_fn)
print("ELU:")
test(test_dataloader, model_elu, loss_fn)