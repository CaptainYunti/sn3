import torch
from load_data import test_data
import matplotlib.pyplot as plt


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Angle Boot",
}

figure = plt.figure(figsize=(16,8))
cols, rows = 6, 3
for i in range(1, cols * rows + 1):
    sample_index = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[sample_index]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="pink")
plt.show()