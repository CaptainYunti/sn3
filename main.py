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

epochs = 10

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

models = [model_128, model_256, model_512, model_512_256, model_512_512, model_512_512_128, model_512_512_512]


loss_fn = nn.CrossEntropyLoss()
optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in models]

print("Learning ...")

for indx, model in enumerate(models):
    print(f"Model: {model.extra_repr()}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------------")
        train(train_dataloader, model, loss_fn, optimizers[indx], device)
        test(test_dataloader, model, loss_fn, device)
        print("Done!\n\n")


torch.save(model_128.state_dict(), "models/model_128_relu_10.pth")
torch.save(model_256.state_dict(), "models/model_256_relu_10.pth")
torch.save(model_512.state_dict(), "models/model_512_relu_10.pth")

torch.save(model_512_256.state_dict(), "models/model_512_256_relu_10.pth")
torch.save(model_512_512.state_dict(), "models/model_512_512_relu_10.pth")

torch.save(model_512_512_128.state_dict(), "models/model_512_512_128_relu_10.pth")
torch.save(model_512_512_512.state_dict(), "models/model_512_512_512_relu_10.pth")



epochs = 30

print("Learning ...")

for indx, model in enumerate(models):
    print(f"Model: {model.extra_repr()}")
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------------")
        train(train_dataloader, model, loss_fn, optimizers[indx], device)
        test(test_dataloader, model, loss_fn, device)
        print("Done!\n\n")


torch.save(model_512.state_dict(), "models/model_512_relu_30.pth")
torch.save(model_512_512.state_dict(), "models/model_512_512_relu_30.pth")
torch.save(model_512_512_128.state_dict(), "models/model_512_512_128_relu_30.pth")