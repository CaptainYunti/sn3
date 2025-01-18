import torch
from torch import nn
from torch.utils.data import DataLoader
import my_models
from load_data import training_data, test_data

BATCH_SIZE = 64


train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = (
    "cuda" 
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")
