import torch 

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.layers = []

    def add_layer(self, layer_name, layer):
        self.add_module(layer_name, layer)
        self.layers.append(layer)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x





