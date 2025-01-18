import torch
import torchvision.models as models

model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), 'learning/model_weights.pth')

model = models.vgg16()

model.load_state_dict(torch.load("learning/model_weights.pth", weights_only=True))
model.eval()

print(model)