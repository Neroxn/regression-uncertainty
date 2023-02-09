import torch
import numpy as np
import tqdm

torch.autograd.set_detect_anomaly(True)

class BaseNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes = [16, 32]):
        super(BaseNetwork, self).__init__()

        # hidden layers
        self.real_layer_sizes = [input_size] + layer_sizes

        for i in range(len(self.real_layer_sizes) - 1):
            setattr(self, 'fc%d' % (i + 1), torch.nn.Linear(self.real_layer_sizes[i], self.real_layer_sizes[i + 1]))

    def forward(self, x):
        for i in range(len(self.real_layer_sizes) - 1):
            x = getattr(self, 'fc%d' % (i + 1))(x)
            x = torch.nn.functional.relu(x)
        return x

class VarianceNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(VarianceNetwork, self).__init__()
        self.network = BaseNetwork(input_size, layer_sizes)

        # get network layers last layer
        self.mu = torch.nn.Linear(layer_sizes[-1], output_size)
        self.sigma = torch.nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        x = self.network(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

def create_multiple_networks(num_networks, input_size, layer_sizes, output_size, **kwargs):
    """
    Create multiple networks with the same architecture
    """
    lr = kwargs.get('lr', 0.001)
    networks = []
    optimizers = []

    for i in range(num_networks):
        networks.append(VarianceNetwork(input_size, layer_sizes, output_size))
        optimizers.append(torch.optim.Adam(networks[i].parameters(), lr=lr))
    return networks, optimizers

def print_networks(networks):
    for k, network in enumerate(networks):
        print('Network %d' % k)
        print(network)

def sig_positive(x):
    """
    Transform the sigma output to the more stable/positive version.
    Args:
        x (torch.Tensor): sigma output

    Returns:
        torch.Tensor: transformed sigma output
    """
    return torch.log(1 + torch.exp(x)) + 1e-6

def train_multiple_networks(
    train_dataloader,
    val_dataloader,
    networks,
    optimizers,
    device,
    **kwargs):
    """
    Train ensamble of networks.
    Args:
        train_dataloader (torch.utils.data.DataLoader): dataloader for training
        val_dataloader (torch.utils.data.DataLoader): dataloader for validation
        networks (list): list of networks
        optimizers (list): list of optimizers
        num_iter (int): number of iterations to train
    """

    # kwargs that controls the training
    num_iters = kwargs.get('num_iter', 1000)
    print_every = kwargs.get('print_every', 100)
    batch_size = kwargs.get('batch_size', 32)
    logger = kwargs.get('logger', None)
    weighted = kwargs.get('weighted', False)
    ds_stats = kwargs.get('ds_stats', None)

    num_networks = len(networks)
    out_mu = np.zeros((batch_size, num_networks))
    out_sig = np.zeros((batch_size, num_networks))
    loss_train = np.zeros((num_networks))
    mse_train = np.zeros((num_networks))

    for k in range(num_networks):
        networks[k] = networks[k].to(device)
    for num_iter in tqdm.tqdm(range(num_iters), position=0, leave=True):
        test_batch_x, test_batch_y = next(iter(val_dataloader)) # this effectively samples a random batch

        for k in range(num_networks):
            # move data to device
            batch_x, batch_y = next(iter(train_dataloader)) # sample random batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # get the network prediction for the training data
            mu_train, sigma_train = networks[k](batch_x)

            sigma_train_pos = sig_positive(sigma_train)

            if weighted is True:
                weights = (1 - torch.tanh(sigma_train_pos)).detach() # we want high variance to have low weight
                loss = torch.mean(
                    (0.5*torch.log(sigma_train_pos) + 0.5*(torch.square(batch_y - mu_train))/sigma_train_pos)*weights
                    ) + 5
            else:
                loss = torch.mean(
                    (0.5*torch.log(sigma_train_pos) + 0.5*(torch.square(batch_y - mu_train))/sigma_train_pos)
                    ) + 5


            if torch.isnan(loss):
                raise ValueError('Loss is NaN')

            optimizers[k].zero_grad()
            loss.backward()
            optimizers[k].step()

            loss_train[k] += loss.item()

            mu_train = mu_train.detach()
            sigma_train = sigma_train.detach()

            if ds_stats is not None:
                y_mu, y_sig = ds_stats[1][0], ds_stats[1][1]
                batch_y_rescaled = batch_y*y_mu + y_sig
                mu_train_rescaled = mu_train*y_mu + y_sig
                mse_train[k] = torch.mean(torch.square(batch_y_rescaled - mu_train_rescaled)).detach().item()
            else:
                mse_train[k] = torch.mean(torch.square(batch_y - mu_train)).detach().item()

            # get the network prediction for the test data
            with torch.no_grad():
                mu_test, sigma_test = networks[k](test_batch_x)
                sigma_test_pos = sig_positive(sigma_test)
                out_mu[:, k] = mu_test.detach().numpy().flatten()
                out_sig[:, k] = sigma_test_pos.detach().numpy().flatten()


        out_mu_final = np.mean(out_mu, axis=1)
        out_sig_final = np.sqrt(np.mean(out_sig, axis=1) + np.mean(np.square(out_mu), axis = 1) - np.square(out_mu_final))
        test_batch_y_rescaled = test_batch_y


        #TODO : Handle this better!
        if ds_stats is not None:
            y_mu, y_sig = ds_stats[1][0], ds_stats[1][1]
            test_batch_y_rescaled = test_batch_y*y_sig + y_mu
            out_mu_final = np.mean(out_mu, axis=1)*y_sig + y_mu


        if logger is not None:
            logger.log_metric(
                metric_name = f"loss",
                metric_value = np.mean(loss_train),
                step = num_iter)

            logger.log_metric(
                metric_name = f"mse_mu",
                metric_value = np.mean(mse_train),
                step = num_iter)

            logger.log_metric(
                metric_name = f"mse_final",
                metric_value = np.mean(np.square(out_mu_final - test_batch_y_rescaled.numpy().flatten())),
                step = num_iter)

        loss_train = np.zeros((num_networks))
        mse_train = np.zeros((num_networks))

    return networks

def test_multiple_networks(x_sample,networks, device, **kwargs):
    """
    Test the ensamble of networks.
    Args:
        test_dataloader (torch.utils.data.DataLoader): dataloader for testing
        networks (list): list of networks

    Returns:
        np.ndarray: predicted mean array
        np.ndarray: predicted standard deviation array
    """
    logger = kwargs.get('logger', None)

    num_networks = len(networks)
    

    x_sample_tensor = torch.from_numpy(x_sample).float().to(device)

    # output for ensemble network
    out_mu_sample  = np.zeros([x_sample_tensor.shape[0], num_networks])
    out_sig_sample = np.zeros([x_sample_tensor.shape[0], num_networks])

    # output for single network
    out_mu_single  = np.zeros([x_sample_tensor.shape[0], 1])
    out_sig_single = np.zeros([x_sample_tensor.shape[0], 1])

    for i in range(num_networks):
        # move network to device
        networks[i].to(device)

        with torch.no_grad():
            mu, sigma = networks[i](x_sample_tensor)
            sigma_pos = sig_positive(sigma)

            out_mu_sample[:,i]  = np.reshape(mu  , (x_sample_tensor.shape[0]))
            out_sig_sample[:,i] = np.reshape(sigma_pos, (x_sample_tensor.shape[0]))

            out_mu_single[:,0]  = np.reshape(mu[:,0], (x_sample_tensor.shape[0]))
            out_sig_single[:,0] = np.reshape(sigma_pos[:,0], (x_sample_tensor.shape[0]))
    
    out_mu_sample_final  = np.mean(out_mu_sample, axis = 1)
    out_sig_sample_aleatoric = np.sqrt(np.mean(out_sig_sample, axis=1)) # model uncertainty
    out_sig_sample_epistemic = np.sqrt(np.mean(np.square(out_mu_sample), axis = 1) - np.square(out_mu_sample_final)) # data uncertainty

    return out_mu_sample_final, out_sig_sample_aleatoric + out_sig_sample_epistemic
