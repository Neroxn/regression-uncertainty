import torch
import numpy as np
import tqdm
import estimations.base
import utils.logger
import transforms.base
from typing import Tuple

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

class MeanNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(MeanNetwork, self).__init__()
        self.network = BaseNetwork(input_size, layer_sizes)

        # get network layers last layer
        self.mu = torch.nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        x = self.network(x)
        mu = self.mu(x)
        return mu

class EnsembleEstimator(estimations.base.UncertaintyEstimator):
    def __init__(self, network_config, optimizer_config):
        self.input_size = network_config.pop("input_size")
        self.output_size = network_config.pop("output_size")
        self.num_networks = network_config.get("num_networks", 5)

        self.optimizer_config = optimizer_config.copy()
        self.network_config = network_config.copy()
        
        self.networks = None
        self.optimizers = None
        self.estimators = None

    def __str__(self) -> str:
        return f"EnsembleEstimator(network_config={self.network_config}, optimizer_config={self.optimizer_config})"

    def init_estimator(self, estimator_class = VarianceNetwork):
        """
        Initialize estimator by using the network and optimizer config
        """
        networks = []
        optimizers = []

        # set network default values
        num_networks = self.num_networks
        layer_sizes = self.network_config.get("layer_sizes", [16, 32])

        optimizer = self.optimizer_config.pop("class", "Adam") # map to optimizers later
        for i in range(num_networks):
            networks.append(VarianceNetwork(self.input_size, layer_sizes, self.output_size))
            optimizers.append(torch.optim.Adam(networks[i].parameters(), **self.optimizer_config))
        
        self.networks = networks
        self.optimizers = optimizers

    def init_predictor(self, predictor_class = MeanNetwork):
        """
        Initialize predictor by using the network and optimizer config
        """
        networks = []
        optimizers = []

        # set network default values
        num_networks = 1
        layer_sizes = self.network_config.get("layer_sizes", [16, 32])

        optimizer = self.optimizer_config.pop("class", "Adam") # map to optimizers later
        for i in range(num_networks):
            networks.append(VarianceNetwork(self.input_size, layer_sizes, self.output_size))
            optimizers.append(torch.optim.Adam(networks[i].parameters(), **self.optimizer_config))
        
        self.networks = networks
        self.optimizers = optimizers
        

    def train_estimator(self, **kwargs):
        self.init_estimator()
        return self.train_multiple_estimator(**kwargs)

    def init_predictor(self, **kwargs):
        self.estimators = self.networks
        self.init_estimator(estimator_class=MeanNetwork)

    def train_predictor(self, **kwargs):
        self.init_predictor()
        return self.train_multiple_estimator(**kwargs)

    def _calculate_uncertainity(self, x, weight_type = "both"):
        """
        Calculate weights for each data point in the batch
        """
        if self.estimators is None:
            raise ValueError("Estimators not trained yet. Please train estimators first.")
        
        out_mu = torch.zeros((x.shape[0], len(self.estimators)))
        out_sig = torch.zeros((x.shape[0], len(self.estimators)))
        with torch.no_grad():
            for i,estimator in enumerate(self.estimators):
                mu,sigma = estimator(x)
                out_mu[:,i] = mu.squeeze()
                out_sig[:,i] = sig_positive(sigma.squeeze())

        # calculate weights using the variance
        out_mu_final  = torch.mean(out_mu, axis = 1).reshape(-1,1)
        
        out_sig_sample_aleatoric = torch.mean(out_sig, axis=1) # model uncertainty
        out_sig_sample_epistemic = torch.mean(torch.square(out_mu), axis = 1) - torch.square(out_mu_final) # data uncertainty
    
        if weight_type == "both":
            return torch.sqrt(out_sig_sample_aleatoric + out_sig_sample_epistemic)
        elif weight_type == "aleatoric":
            return torch.sqrt(out_sig_sample_aleatoric)
        elif weight_type == "epistemic":
            return torch.sqrt(out_sig_sample_epistemic)
        else:
            raise ValueError("Weight type not supported. Please choose from aleatoric, epistemic, both")

    def _calculate_weights(self, x, weight_type = "aleatoric"):
        uncertainity = self._calculate_uncertainity(x, weight_type = weight_type)
        weights = torch.tanh(uncertainity)
        return uncertainity

    def train_multiple_estimator(
        self,
        train_config : dict,
        train_dl : torch.utils.data.DataLoader,
        val_dl : torch.utils.data.DataLoader,
        device : torch.device,
        transforms : Tuple[transforms.base.Transform],
        logger : utils.logger.NetworkLogger,
        logger_prefix : str = "estimator",
        weighted_training = False) -> None:
        
        num_iters = train_config.get("num_iter", 1000)
        val_every = train_config.get('val_every', 250)
        print_every = train_config.get('print_every', 50)
        train_type = train_config.get('train_type', 'iter')

        if train_type == "epoch":
            num_iters = num_iters * len(train_dl)
            val_every = val_every * len(train_dl)
            print_every = print_every * len(train_dl)

        x_transform,y_transform = transforms
        networks = self.networks
        optimizers = self.optimizers
        num_networks = len(networks)

        loss_train = torch.zeros((num_networks))
        mse_train = torch.zeros((num_networks))

        for k in range(num_networks):
            networks[k] = networks[k].to(device)

        # for epoch based training, create iterator per network
        if train_type == 'epoch':
            train_iters = []
            for k in range(num_networks):
                train_iters.append(iter(train_dl))

        for num_iter in tqdm.tqdm(range(num_iters), position=0, leave=True):
            for k in range(num_networks):
                # move data to device
                if train_type == 'iter':
                    batch_x, batch_y = next(iter(train_dl))
                elif train_type == 'epoch':
                    try:
                        batch_x, batch_y = next(train_iters[k]) # sample random batch
                    except StopIteration:
                        train_iters[k] = iter(train_dl)
                        batch_x, batch_y = next(train_iters[k])
                else:
                    raise ValueError("Train type not supported. Please choose from iter, epoch")
                
                batch_x, batch_y = x_transform.forward(batch_x), y_transform.forward(batch_y)

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # get the network prediction for the training data
                mu_train, sigma_train = networks[k](batch_x)
                sigma_train_pos = sig_positive(sigma_train)

                if weighted_training is True:
                    weights = self._calculate_weights(batch_x)
                    loss = torch.mean(
                        (0.5*torch.log(sigma_train_pos) + 0.5*(torch.square(batch_y - mu_train))/sigma_train_pos)*weights
                        + 5) 
                else:
                    loss = torch.mean(
                        (0.5*torch.log(sigma_train_pos) + 0.5*(torch.square(batch_y - mu_train))/sigma_train_pos)
                        + 5) 

                if torch.isnan(loss):
                    raise ValueError('Loss is NaN')

                # calculate the loss and update the weights
                optimizers[k].zero_grad()
                loss.backward()
                optimizers[k].step()

                loss_train[k] += loss.to('cpu').item()
                mu_train = mu_train.detach()
                sigma_train = sigma_train.detach()
                
                batch_y_transformed = y_transform.backward(batch_y)
                mu_train_transformed = y_transform.backward(mu_train)

                mse_train[k] += torch.mean(
                    torch.square(batch_y_transformed - mu_train_transformed)
                    ).to('cpu')
                
            logger.log_metric(
                metric_dict = {
                    f"{logger_prefix}_loss" : torch.mean(loss_train),
                    f"{logger_prefix}_mse_mu" : torch.mean(mse_train),
                    f"{logger_prefix}_rmse_mu" : torch.sqrt(torch.mean(mse_train)),
                },
                step = (num_iter + 1),
                print_log = (num_iter + 1) % print_every == 0
            )

            if (num_iter + 1) % val_every == 0:
                val_mu, _, val_true = self.evaluate_multiple_networks(val_dl, networks, transforms, device, logger = logger)
                logger.log_metric(
                    metric_dict = {
                        f"{logger_prefix}_val_mse_mean" : torch.mean(torch.square(val_mu - val_true)),
                        f"{logger_prefix}_val_rmse_mean" : torch.sqrt(torch.mean(np.square(val_mu - val_true)))
                    },
                    step = (num_iter + 1),
                )   
            
            loss_train = torch.zeros((num_networks))
            mse_train = torch.zeros((num_networks))
 
        return networks

    def evaluate_multiple_networks(self, val_dataloader, networks, transforms, device, **kwargs):
        """
        Evaluate the ensamble of networks.
        Args:
            val_dataloader (torch.utils.data.DataLoader): dataloader for testing
            networks (list): list of networks

        Returns:
            np.ndarray: predicted mean array
            np.ndarray: predicted standard deviation array
        """
        logger = kwargs.get('logger', None)
        num_networks = len(networks)
        
        # output for ensemble network
        out_mu  = torch.zeros([len(val_dataloader.dataset), num_networks])
        out_sig = torch.zeros([len(val_dataloader.dataset), num_networks])

        batch_y_true = torch.zeros([len(val_dataloader.dataset), 1])

        x_transform, y_transform = transforms

        current_index = 0

        for i, (batch_x, batch_y) in enumerate(val_dataloader):
            batch_x = x_transform.forward(batch_x)
            batch_y_true[current_index: current_index + batch_y.shape[0],0] = batch_y.reshape((batch_y.shape[0])).cpu()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            for j in range(num_networks):
                # move network to device
                networks[j].to(device)
                with torch.no_grad():
                    mu, sigma = networks[j](batch_x)
                    sigma_pos = sig_positive(sigma)            

                    out_mu[current_index:current_index + mu.shape[0],j] = mu.reshape((mu.shape[0])).cpu()
                    out_sig[current_index: current_index + mu.shape[0],j] = sigma_pos.reshape((mu.shape[0])).cpu()

            current_index += mu.shape[0]

        out_mu = y_transform.backward(out_mu)
        out_mu_final  = torch.mean(out_mu, axis = 1).reshape(-1,1)

        out_sig_sample_aleatoric = torch.sqrt(torch.mean(out_sig, axis=1)) # model uncertainty
        out_sig_sample_epistemic = torch.sqrt(torch.mean(np.square(out_mu), axis = 1) - torch.square(out_mu_final)) # data uncertainty
        # VAR X = E[X^2] - E[X]^2 
        
        return out_mu_final, out_sig_sample_aleatoric + out_sig_sample_epistemic, batch_y_true

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

