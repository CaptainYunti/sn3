import torch
from torch import nn
from torch.utils.data import DataLoader
import my_models
from load_data import training_data, test_data
from train_test import train, test

BATCH_SIZE = 64
#EPOCHS = 30


train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = (
    "cuda" 
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}\n")

model_128 = my_models.OneHiddenLayer(128).to(device)
model_256 = my_models.OneHiddenLayer(256).to(device)
model_512 = my_models.OneHiddenLayer(512).to(device)

model_512_256 = my_models.TwoHiddenLayer((512, 256)).to(device)
model_512_512 = my_models.TwoHiddenLayer((512, 512)).to(device)

model_512_512_512 = my_models.ThreeHiddenLayer((512,512,512)).to(device)
model_512_512_128 = my_models.ThreeHiddenLayer((512,512,128)).to(device)

#models = [model_128, model_256, model_512, model_512_512, model_512_256]
#models = [model_512_512, model_512_256]
#models = [model_512_512_512, model_512_512_128]
#models = [model_512, model_512_512, model_512_512_128]



epochs = 10

models = []

#models = [model_128, model_256, model_512, model_512_256, model_512_512, model_512_512_128, model_512_512_512]
#models = [model_512_512_512]
loss_fn = nn.CrossEntropyLoss()
optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in models]



print("Learning (10) ...")

for indx, model in enumerate(models):
    print(f"Model: {model.extra_repr()}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------------")
        train(train_dataloader, model, loss_fn, optimizers[indx], device)
        test(test_dataloader, model, loss_fn, device)
        print("Done!\n\n")

if len(models) > 0:
    torch.save(model_128.state_dict(), "models/model_128_relu_sgd_10.pth")
    torch.save(model_256.state_dict(), "models/model_256_relu_sgd_10.pth")
    torch.save(model_512.state_dict(), "models/model_512_relu_sgd_10.pth")

    torch.save(model_512_256.state_dict(), "models/model_512_256_relu_sgd_10.pth")
    torch.save(model_512_512.state_dict(), "models/model_512_512_relu_sgd_10.pth")

    torch.save(model_512_512_128.state_dict(), "models/model_512_512_128_relu_sgd_10.pth")
    torch.save(model_512_512_512.state_dict(), "models/model_512_512_512_relu_sgd_10.pth")



epochs = 30

models = []
#models = [model_512, model_512_512, model_512_512_128]
optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in models]

print("Learning (30) ...")

for indx, model in enumerate(models):
    print(f"Model: {model.extra_repr()}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------------")
        train(train_dataloader, model, loss_fn, optimizers[indx], device)
        test(test_dataloader, model, loss_fn, device)
        print("Done!\n\n")


if len(models) > 0:
    torch.save(model_512.state_dict(), "models/model_512_relu_sgd_30.pth")
    torch.save(model_512_512.state_dict(), "models/model_512_512_relu_sgd_30.pth")
    torch.save(model_512_512_128.state_dict(), "models/model_512_512_128_relu_sgd_30.pth")




epochs = 40

model_512_sgd =  my_models.OneHiddenLayer(512).to(device)
model_512_adadelta = my_models.OneHiddenLayer(512).to(device)
model_512_adam = my_models.OneHiddenLayer(512).to(device)

models = []
#models = [model_512_sgd, model_512_adadelta, model_512_adam]

# optimizers = [
#     torch.optim.SGD(model_512_sgd.parameters()),
#     torch.optim.Adadelta(model_512_adadelta.parameters()),
#     torch.optim.Adam(model_512_adam.parameters())
#     ]

print("Learning (40) ...")

for indx, model in enumerate(models):
    print(f"Model: {model.extra_repr()}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------------")
        train(train_dataloader, model, loss_fn, optimizers[indx], device)
        test(test_dataloader, model, loss_fn, device)
        print("Done!\n\n")


if len(models) > 0:
    torch.save(model_512_sgd.state_dict(), "models/model_512_relu_sgd_40.pth")
    torch.save(model_512_adadelta.state_dict(), "models/model_512_relu_adadelta_40.pth")
    torch.save(model_512_adam.state_dict(), "models/model_512_relu_adam_40.pth")





epochs = 20

model_512_adam_relu = my_models.OneHiddenLayer(512).to(device)
model_512_adam_sigmoid = my_models.OneHiddenLayer(512, nn.Sigmoid).to(device)
model_512_adam_tanh = my_models.OneHiddenLayer(512, nn.Tanh).to(device)
model_512_adam_elu = my_models.OneHiddenLayer(512, nn.ELU).to(device)

models = [model_512_adam_relu, model_512_adam_sigmoid, model_512_adam_tanh, model_512_adam_elu]
#models = []

optimizers = [torch.optim.Adam(model.parameters()) for model in models]


print("Learning (16) ...")

for indx, model in enumerate(models):
    print(f"Model: {model.extra_repr()}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------------")
        train(train_dataloader, model, loss_fn, optimizers[indx], device)
        test(test_dataloader, model, loss_fn, device)
        print("Done!\n\n")


# if len(models) > 0:
#     torch.save(model_512_adam_relu.state_dict(), "models/model_512_relu_adam_16.pth")
#     torch.save(model_512_adam_sigmoid.state_dict(), "models/model_512_sigmoid_adam_16.pth")
#     torch.save(model_512_adam_tanh.state_dict(), "models/model_512_tanh_adam_16.pth")
#     torch.save(model_512_adam_elu.state_dict(), "models/model_512_elu_adam_16.pth")