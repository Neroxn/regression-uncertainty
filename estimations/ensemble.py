import torch
import numpy as np
import tqdm
import estimations.base
import utils.logger
import transforms.base
from typing import Tuple

torch.autograd.set_detect_anomaly(True)
class EnsembleEstimator(estimations.base.UncertaintyEstimator):
    def __init__(self, network_config, optimizer_config):
        self.num_networks = network_config.get("num_networks", 5)

        self.optimizer_config = optimizer_config.copy()
        self.network_config = network_config.copy()
        
        self.optimizers = None
        self.estimators = None

    def __str__(self) -> str:
        return f"EnsembleEstimator(network_config={self.network_config}, optimizer_config={self.optimizer_config})"

    def init_estimator(self, network_name : str):
        """
        Initialize estimator by using the network and optimizer config
        """
        networks = []
        optimizers = []

        num_networks = self.num_networks
        for i in range(num_networks):
            estimator = self._build_network(self.network_config.copy(), network_name)
            optimizer = self._build_optimizer(self.optimizer_config, estimator.parameters())

            networks.append(estimator)
            optimizers.append(optimizer)
        
        self.estimators = networks
        self.estimators_optimizer = optimizers

    def init_predictor(self, network_name : str):
        """
        Initialize predictor by using the network and optimizer config
        """
        self.predictor = self._build_network(self.network_config, network_name)
        self.predictor_optimizer = self._build_optimizer(self.optimizer_config, self.predictor.parameters())
        

    def train_estimator(self, **kwargs):
        self.init_estimator(network_name = "estimator_network")
        return self.train_multiple_estimator(**kwargs)

    def train_predictor(self, weighted_training, **kwargs):
        self.init_predictor(network_name = "predictor_network")
        return self.train_single_predictor(weighted_training=weighted_training, **kwargs)

    def _calculate_uncertainity(self, x, y, weight_type = "both"):
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
                out_sig[:,i] = sigma.squeeze()

        # calculate weights using the variance
        out_mu_final  = torch.mean(out_mu, axis = 1).reshape(-1,1)
        
        out_sig_sample_aleatoric = torch.mean(out_sig, axis=1) # model uncertainty
        out_sig_sample_epistemic = torch.mean(torch.square(out_mu), axis = 1) - torch.square(out_mu_final)  # data uncertainty
    
        # replace 0 for all the elements in out_sig_sample_epistemic that is less than 0
        out_sig_sample_epistemic[out_sig_sample_epistemic < 0] = 0

        if weight_type == "both":
            out_sig_pred =  torch.sqrt(out_sig_sample_aleatoric + out_sig_sample_epistemic)
        elif weight_type == "aleatoric":
            out_sig_pred = torch.sqrt(out_sig_sample_aleatoric)
        elif weight_type == "epistemic":
            out_sig_pred =  torch.sqrt(out_sig_sample_epistemic)
        else:
            raise ValueError("Weight type not supported. Please choose from aleatoric, epistemic, both")
        
        return out_sig_pred

    def _calculate_weights(self, x, y, weight_type = "aleatoric"):
        weights = self._calculate_uncertainity(x, y, weight_type = weight_type)
        return weights

    def train_multiple_estimator(
        self,
        train_config : dict,
        train_dl : torch.utils.data.DataLoader,
        val_dl : torch.utils.data.DataLoader,
        device : torch.device,
        transforms : Tuple[transforms.base.Transform],
        logger : utils.logger.NetworkLogger,
        logger_prefix : str = "estimator") -> None:
        
        num_iters = train_config.get("num_iter", 1000)
        val_every = train_config.get('val_every', num_iters)
        print_every = train_config.get('print_every', num_iters + 1)
        train_type = train_config.get('train_type', 'iter')

        if train_type == "epoch":
            num_iters = num_iters * len(train_dl)
            val_every = val_every * len(train_dl)
            print_every = print_every * len(train_dl)

        _,y_transform = transforms
        networks = self.estimators
        optimizers = self.estimators_optimizer
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
                
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # get the network prediction for the training data
                mu_train, sigma_train = networks[k](batch_x)

                loss = torch.mean(
                    (0.5*torch.log(sigma_train) + 0.5*(torch.square(batch_y - mu_train))/sigma_train)
                    + 5) 
                    
                #mse loss
                loss = torch.mean(
                    torch.square(batch_y - mu_train)
                    )
                if torch.isnan(loss):
                    raise ValueError('Loss is NaN')

                # calculate the loss and update the weights
                optimizers[k].zero_grad()
                loss.backward()

                # clamp gradients that are too big
                torch.nn.utils.clip_grad_norm_(networks[k].parameters(), 30.0)

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

        _, y_transform = transforms

        current_index = 0

        for batch in tqdm.tqdm(val_dataloader):
            if len(batch) == 2:
                batch_x, batch_y = batch
            elif len(batch) == 1:
                batch_x = batch[0]
                batch_y = torch.zeros((batch_x.shape[0], 1))
            batch_y_true[current_index: current_index + batch_y.shape[0],0] = batch_y.reshape((batch_y.shape[0])).cpu()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            for j in range(num_networks):
                # move network to device
                networks[j].to(device)
                with torch.no_grad():
                    mu, sigma = networks[j](batch_x)         

                    out_mu[current_index:current_index + mu.shape[0],j] = mu.reshape((mu.shape[0])).cpu()
                    out_sig[current_index: current_index + mu.shape[0],j] = sigma.reshape((mu.shape[0])).cpu()

            current_index += mu.shape[0]

        out_mu = y_transform.backward(out_mu)
        out_mu_final  = torch.mean(out_mu, axis = 1).reshape(-1,1)

        out_sig_sample_aleatoric = torch.sqrt(torch.mean(out_sig, axis=1)) # model uncertainty
        out_sig_sample_epistemic = torch.sqrt(torch.mean(np.square(out_mu), axis = 1) - torch.square(out_mu_final)) # data uncertainty
        # VAR X = E[X^2] - E[X]^2 
        
        return out_mu_final, out_sig_sample_aleatoric + out_sig_sample_epistemic, batch_y_true
    


    def evaluate_predictor(self, val_dataloader, predictor, transforms, device, **kwargs):
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
        
        # output for ensemble network
        out_mu  = torch.zeros([len(val_dataloader.dataset)])

        batch_y_true = torch.zeros([len(val_dataloader.dataset), 1])

        x_transform, y_transform = transforms

        current_index = 0

        # move network to device
        predictor.to(device)
        
        for batch in tqdm.tqdm(val_dataloader):
            if len(batch) == 2:
                batch_x, batch_y = batch
            elif len(batch) == 1:
                batch_x = batch[0]
                batch_y = torch.zeros((batch_x.shape[0], 1))
            batch_y_true[current_index: current_index + batch_y.shape[0],0] = batch_y.reshape((batch_y.shape[0])).cpu()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            with torch.no_grad():
                mu = predictor(batch_x)         
                out_mu[current_index:current_index + mu.shape[0]] = mu.reshape((mu.shape[0])).cpu()

            current_index += mu.shape[0]

        out_mu = y_transform.backward(out_mu)
        out_mu_final  = out_mu.reshape(-1,1)

        return out_mu_final, batch_y_true


    def train_single_predictor(
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
            val_every = train_config.get('val_every', num_iters)
            print_every = train_config.get('print_every', num_iters + 1)
            train_type = train_config.get('train_type', 'iter')
            weight_type = train_config.get('weight_type', 'both')

            if train_type == "epoch":
                num_iters = num_iters * len(train_dl)
                val_every = val_every * len(train_dl)
                print_every = print_every * len(train_dl)

            x_transform,y_transform = transforms

            predictor = self.predictor.to(device)
            optimizer = self.predictor_optimizer

            iter_dl = iter(train_dl)
            for num_iter in tqdm.tqdm(range(num_iters), position=0, leave=True):
                # move data to device
                if train_type == 'iter':
                    batch_x, batch_y = next(iter(train_dl))
                elif train_type == 'epoch':
                    try:
                        batch_x, batch_y = next(iter_dl) # sample random batch
                    except StopIteration:
                        batch_x, batch_y = next(iter(train_dl))
                else:
                    raise ValueError("Train type not supported. Please choose from iter, epoch")
                

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                mu_train = predictor(batch_x) # predict mean
                # use mse as the loss
                if weighted_training:
                    weights = self._calculate_weights(batch_x,batch_y, weight_type = weight_type).detach().to(device)
                    loss = torch.mean(torch.square(mu_train - batch_y) * weights)
                else:
                    loss = torch.mean(torch.square(mu_train - batch_y))
                    
                if torch.isnan(loss):
                    raise ValueError('Loss is NaN')

                # calculate the loss and update the weights
                optimizer.zero_grad()
                loss.backward()

                # clamp gradients that are too big
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)

                optimizer.step()

                loss_train = loss.to('cpu').item()
                mu_train = mu_train.detach()
                
                batch_y_transformed = y_transform.backward(batch_y)
                mu_train_transformed = y_transform.backward(mu_train)

                mse_train = torch.mean(
                    torch.square(batch_y_transformed - mu_train_transformed)
                    ).to('cpu')
                    
                logger.log_metric(
                    metric_dict = {
                        f"{logger_prefix}_loss" : loss_train,
                        f"{logger_prefix}_mse_mu" : mse_train,
                        f"{logger_prefix}_rmse_mu" : torch.sqrt(mse_train),
                    },
                    step = (num_iter + 1),
                    print_log = (num_iter + 1) % print_every == 0
                )

                if (num_iter + 1) % val_every == 0:
                    val_mu, val_true = self.evaluate_predictor(val_dl, predictor, transforms, device, logger = logger)
                    logger.log_metric(
                        metric_dict = {
                            f"{logger_prefix}_val_mse_mean" : torch.mean(torch.square(val_mu - val_true)),
                            f"{logger_prefix}_val_rmse_mean" : torch.sqrt(torch.mean(torch.square(val_mu - val_true)))
                        },
                        step = (num_iter + 1),
                    )   
                
                loss_train = 0 
                mse_train = 0 
    
            return predictor

