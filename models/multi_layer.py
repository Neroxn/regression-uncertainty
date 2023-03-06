import torch

class LinearVarianceNetworkHead(torch.nn.Module):
    """ A simple VarianceNetwork head for MLP"""
    def __init__(self, in_features, out_features):
        super(LinearVarianceNetworkHead, self).__init__()
        self.mu = torch.nn.Linear(in_features, out_features)
        self.log_var = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = torch.log(1 + torch.exp(self.log_var(x))) + 1e-6
        return mu, log_var
    

class Conv2DVarianceNetworkHead(torch.nn.Module):
    """ A simple VarianceNetwork head for CNN"""
    def __init__(self, in_features, out_features):
        super(Conv2DVarianceNetworkHead, self).__init__()
        self.mu = torch.nn.Conv2d(in_features, out_features, 1)
        self.log_var = torch.nn.Conv2d(in_features, out_features, 1)

    def forward(self, x):
        mu = self.mu(x)
        log_var = torch.log(1 + torch.exp(self.log_var(x))) + 1e-6
        return mu, log_var

