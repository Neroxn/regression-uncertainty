import torch

class LinearVarianceNetworkHead(torch.nn.Module):
    """ A simple VarianceNetwork head for MLP"""
    def __init__(self, in_features, out_features):
        super(LinearVarianceNetworkHead, self).__init__()
        self.mu = torch.nn.Linear(in_features, out_features)
        self.log_var = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        log_var = torch.clamp(log_var, max=15) # e^
        log_var = torch.log(1 + torch.exp(log_var)) + 1e-6
        return mu, log_var
    

class Conv2DVarianceNetworkHead(torch.nn.Module):
    """ A simple VarianceNetwork head for CNN"""
    def __init__(self, in_channel, out_channnel):
        super(Conv2DVarianceNetworkHead, self).__init__()
        self.mu = torch.nn.Conv2d(in_channel, out_channnel, 1)
        self.log_var = torch.nn.Conv2d(in_channel, out_channnel, 1)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        # clamp log var to avoid numerical instability
        log_var = torch.clamp(log_var, min=-7, max=7) 
        log_var = torch.log(1 + torch.exp(log_var)) + 1e-6
        return mu, log_var

