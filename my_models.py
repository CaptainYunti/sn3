from torch import nn



#jedna warstwa ukryta
class OneHiddenLayer(nn.Module):
    def __init__(self, perceptrons = 256, activation=nn.ReLU):
        super().__init__()
        self.perceptrons = perceptrons
        self.activation = activation
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, self.perceptrons),
            activation(),
            nn.Linear(self.perceptrons, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def extra_repr(self):
        return(f"Hidden layers: 1, number of perceptrons: {self.perceptrons}, activation function: {self.activation}")


#dwie wartswy ukryte
class TwoHiddenLayer(nn.Module):
    def __init__(self, perceptrons = (256, 128), activation=nn.ReLU):
        super().__init__()
        self.perceptrons = perceptrons
        self.activation = activation
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, perceptrons[0]),
            activation(),
            nn.Linear(perceptrons[0], perceptrons[1]),
            activation(),
            nn.Linear(perceptrons[1], 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def extra_repr(self):
        return(f"Hidden layers: {len(self.perceptrons)}, number of perceptrons: {self.perceptrons}, activation function: {self.activation}")



#trzy warstwy ukryte
class ThreeHiddenLayer(nn.Module):
    def __init__(self, perceptrons = (512, 256, 128), activation=nn.ReLU):
        super().__init__()
        self.perceptrons = perceptrons
        self.activation = activation
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, perceptrons[0]),
            activation(),
            nn.Linear(perceptrons[0], perceptrons[1]),
            activation(),
            nn.Linear(perceptrons[1], perceptrons[2]),
            activation(),
            nn.Linear(perceptrons[2], 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def extra_repr(self):
        return(f"Hidden layers: {len(self.perceptrons)}, number of perceptrons: {self.perceptrons}, activation function: {self.activation}")
