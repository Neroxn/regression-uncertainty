import torch
import numpy as np
import tqdm
import estimations.base
import utils.logger
import transforms.base
import utils.metrics
from typing import Tuple
import warnings
import time
import os

def warmup(weights, epoch, max_epoch):
    """
    Warmup function for the weights of the ensemble estimator
    Arguments:
        weights: Weights of the ensemble estimator
        epoch: Current epoch
        max_epoch: Maximum number of epochs
    Returns:
        weights: Updated weights
    """
    if epoch > max_epoch:
        epoch = max_epoch
    alpha =  (1 - np.cos(np.pi * epoch / (2*max_epoch)) ** 2)
    return weights * alpha + (1 - alpha) * np.ones(weights.shape)

torch.autograd.set_detect_anomaly(True)
class EnsembleEstimator(estimations.base.UncertaintyEstimator):
    def __init__(self, network_config, optimizer_config, num_networks = 5):
        self.num_networks = num_networks
        self.scheduler_config = optimizer_config.pop("scheduler", None)
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
        schedulers = []

        num_networks = self.num_networks
        print(f"Number of networks: {num_networks}")
        for i in range(num_networks):
            estimator = self._build_network(self.network_config.copy(), network_name)
            optimizer = self._build_optimizer(self.optimizer_config, estimator.parameters())
            scheduler = self._build_scheduler(self.scheduler_config, optimizer)

            networks.append(estimator)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
        
        self.estimators = networks
        self.estimators_optimizer = optimizers
        self.estimators_scheduler = schedulers

    def init_predictor(self, network_name : str):
        """
        Initialize predictor by using the network and optimizer config
        """

        self.predictor = self._build_network(self.network_config, network_name)
        self.predictor_optimizer = self._build_optimizer(self.optimizer_config, self.predictor.parameters())
        self.predictor_scheduler = self._build_scheduler(self.scheduler_config, self.predictor_optimizer)
        

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
        if weight_type == "both":
            out_sig_var = out_sig_sample_aleatoric + out_sig_sample_epistemic
            out_sig_var[out_sig_var < 0] = 0
            out_sig_pred =  torch.sqrt(out_sig_var) # safety check
        elif weight_type == "aleatoric":
            out_sig_sample_aleatoric[out_sig_sample_aleatoric < 0] = 0
            out_sig_pred = torch.sqrt(out_sig_sample_aleatoric) #safety check
        elif weight_type == "epistemic":
            out_sig_sample_epistemic[out_sig_sample_epistemic < 0] = 0
            out_sig_pred =  torch.sqrt(out_sig_sample_epistemic) #safety check
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
        transforms : dict,
        logger : utils.logger.NetworkLogger,
        logger_prefix : str = "estimator",
        metrics : dict = None,
        categorize : bool = False,
        checkpoint = None
        ) -> None:
        """
        Parameters:
        -----------
        train_config : dict
            Dictionary containing the training configuration
        train_dl : torch.utils.data.DataLoader
            Training data loader
        val_dl : torch.utils.data.DataLoader
            Validation data loader
        device : torch.device
            Device to use for training
        transforms : dict
            Dictionary containing the transforms to be applied on the data
        logger : utils.logger.NetworkLogger
            Logger to log the training and validation metrics
        logger_prefix : str
            Prefix to be used for logging the metrics
        metrics : dict
            Dictionary containing the metrics to be used for training
        categorize : bool
            Whether to categorize the data or not
        """
        
        num_iters = train_config.get("num_iter", 1000)
        val_every = train_config.get('val_every', num_iters)
        print_every = train_config.get('print_every', num_iters + 1)
        train_type = train_config.get('train_type', 'iter')

        if checkpoint:
            start_time = time.strftime("%Y%m%d-%H%M%S")
        # save every epoch 
        save_every = 1
        if train_type == "epoch":
            num_iters = num_iters * len(train_dl)
            val_every = val_every * len(train_dl)
            print_every = print_every * len(train_dl)

        networks = self.estimators
        if len(networks) == 0:
            warnings.warn("No networks found. Skipping the estimator training.")
            return networks

        optimizers = self.estimators_optimizer
        num_networks = len(networks)

        self.device = device
        for k in range(num_networks):
            networks[k] = networks[k].to(device)

        # for epoch based training, create iterator per network
        train_iters = []
        for k in range(num_networks):
            train_iters.append(iter(train_dl))

        for num_iter in tqdm.tqdm(range(num_iters), position=0, leave=True):
            # initialize values for tracking
            loss_train = 0
            batch_sigma_pred = torch.zeros((num_networks, train_dl.batch_size))
            batch_pred = torch.zeros((num_networks, train_dl.batch_size))
            batch_true = torch.zeros((num_networks, train_dl.batch_size))

            for k in range(num_networks):
                # move data to device
                try:
                    batch_x, batch_y = next(train_iters[k]) # sample random batch
                except:
                    train_iters[k] = iter(train_dl)
                    batch_x, batch_y = next(train_iters[k])
                
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()

                # get the network prediction for the training data
                mu_train, sigma_train = networks[k](batch_x)
                loss = torch.mean(
                    (0.5*torch.log(sigma_train) + 0.5*(torch.square(batch_y - mu_train))/sigma_train)
                    + 5) 

                if torch.isnan(loss):
                    raise ValueError('Loss is NaN')

                # calculate the loss and update the weights
                optimizers[k].zero_grad()

                # log(1 + exp(x)) derivative is 
                loss.backward()

                # clamp gradients that are too big
                torch.nn.utils.clip_grad_norm_(networks[k].parameters(), 30.0)

                optimizers[k].step()
                loss_train += loss
                mu_train = mu_train.detach()
                sigma_train = sigma_train.detach()
                
                batch_y_transformed = transforms["train"]["y"].backward(batch_y)
                mu_train_transformed = transforms["train"]["y"].backward(mu_train)

                # also handle the last batches
                batch_pred[k, :batch_x.shape[0]] = mu_train_transformed.reshape(-1)
                batch_true[k, :batch_x.shape[0]] = batch_y_transformed.reshape(-1)
                batch_sigma_pred[k, :batch_x.shape[0]] = sigma_train.reshape(-1)

                if train_type == 'epoch':
                    # if epoch is passed
                    if (num_iter + 1) % len(train_dl) == 0:
                        if self.estimators_scheduler[k] is not None:
                            self.estimators_scheduler[k].step()
                else:
                    if self.estimators_scheduler[k] is not None:
                        self.estimators_scheduler[k].step()
                        
            # log the criterion/loss, current lr
            logger.log_metric(
                metric_dict = {
                logger_prefix + "_train_loss": loss.item(),
                logger_prefix + "_lr": optimizers[0].param_groups[0]['lr'],
                },
                step = (num_iter + 1),
            )

            train_metric_list = metrics.get("train", None)
            if train_metric_list is not None:
                train_average_result = {}
                for k in range(num_networks):
                    y_true = batch_true[k,:batch_x.shape[0]]
                    y_pred = batch_pred[k,:batch_x.shape[0]]
                    sigma_pred = batch_sigma_pred[k,:batch_x.shape[0]]

                    # calculate the metrics
                    train_metric_list.forward(y_pred = y_pred, y_true = y_true, sigma_pred = sigma_pred)
                    train_result = train_metric_list.get_metrics()

                    for key,val in train_result.items():
                        if key not in train_average_result:
                            train_average_result[key] = 0
                        train_average_result[key] += val

                    logger.log_metric(
                        metric_dict = train_metric_list.get_metrics(add_prefix = logger_prefix + f"_train_{k}"),
                        step = (num_iter + 1),
                    )

                # average the metrics
                logger.log_metric(
                    metric_dict = {logger_prefix + "_train_average_" + k: v/num_networks for k,v in train_average_result.items()},
                    step = (num_iter + 1),
                )
                

            if (num_iter + 1) % val_every == 0:
                val_mu, _,  val_true = self.evaluate_multiple_networks(val_dl, networks, transforms, device)
                val_mu = transforms["val"]["y"].backward(val_mu)
                val_true = transforms["val"]["y"].backward(val_true)

                val_metric_list = metrics.get("val", None)
                if val_metric_list is not None:
                    if categorize:
                        assert hasattr(val_dl.dataset, "get_categories"), "Dataset class must have get_categories method"
                        categories = val_dl.dataset.get_categories()

                        # get category labels for all val_true
                        val_true_cat = np.array([categories.get(int(val_true[i]),"zero-shot") for i in range(len(val_true))])
                        for cat in np.unique(val_true_cat):
                            cat_mask = val_true_cat == cat
                            cat_val_true = val_true[cat_mask]
                            cat_val_mu = val_mu[cat_mask]

                            val_metric_list.forward(cat_val_mu, cat_val_true)
                            logger.log_metric(
                                metric_dict = val_metric_list.get_metrics(add_prefix = logger_prefix + "_val_" + cat),
                                step = (num_iter + 1),
                            )
                    val_metric_list.forward(val_mu, val_true)
                    logger.log_metric(
                        metric_dict = val_metric_list.get_metrics(add_prefix = logger_prefix + "_val"),
                        step = (num_iter + 1),
                    )

                if checkpoint is not None:
                    checkpoint_folder = os.path.join(checkpoint, start_time)
                    if not os.path.exists(checkpoint_folder):
                        os.makedirs(checkpoint_folder)
                    
                    for i, network in enumerate(networks):
                        torch.save(network.state_dict(), os.path.join(checkpoint_folder,f"estimator_network_{i}.pth"))
    
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
        num_networks = len(networks)
        
        # output for ensemble network
        out_mu  = torch.zeros([len(val_dataloader.dataset), num_networks])
        out_sig = torch.zeros([len(val_dataloader.dataset), num_networks])

        batch_y_true = torch.zeros([len(val_dataloader.dataset), 1])
        current_index = 0

        for batch in tqdm.tqdm(val_dataloader):
            if len(batch) == 2:
                batch_x, batch_y = batch
            elif len(batch) == 1:
                batch_x = batch[0]
                batch_y = torch.zeros((batch_x.shape[0], 1))
            batch_y_true[current_index: current_index + batch_y.shape[0],0] = batch_y.reshape((batch_y.shape[0])).cpu()
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            for j in range(num_networks):
                # move network to device
                networks[j].to(device)
                with torch.no_grad():
                    mu, sigma = networks[j](batch_x)         
                    out_mu[current_index:current_index + mu.shape[0],j] = mu.reshape((mu.shape[0])).cpu()
                    out_sig[current_index: current_index + mu.shape[0],j] = sigma.reshape((mu.shape[0])).cpu()

            current_index += mu.shape[0]

        out_mu_final  = torch.mean(out_mu, axis = 1).reshape(-1,1)

        out_sig_sample_aleatoric = torch.sqrt(torch.mean(out_sig, axis=1)) # model uncertainty
        out_sig_sample_epistemic = torch.sqrt(torch.mean(np.square(out_mu), axis = 1) - torch.square(out_mu_final)) # data uncertainty
        # VAR X = E[X^2] - E[X]^2 
        
        return out_mu_final, out_sig_sample_aleatoric + out_sig_sample_epistemic, batch_y_true
    

    def evaluate_predictor(self, val_dataloader, predictor, transforms, device,**kwargs):
        """
        Evaluate the ensamble of networks.
        Args:
            val_dataloader (torch.utils.data.DataLoader): dataloader for testing
            networks (list): list of networks

        Returns:
            np.ndarray: predicted mean array
            np.ndarray: predicted standard deviation array
        """
        
        # output for ensemble network
        out_mu  = torch.zeros([len(val_dataloader.dataset)])
        batch_y_true = torch.zeros([len(val_dataloader.dataset), 1])

        current_index = 0
        # move network to device
        predictor.to(device)
        
        for i,batch in tqdm.tqdm(enumerate(val_dataloader)):
            if len(batch) == 2:
                batch_x, batch_y = batch
            elif len(batch) == 1:
                batch_x = batch[0]
                batch_y = torch.zeros((batch_x.shape[0], 1))
            batch_y_true[current_index: current_index + batch_y.shape[0],0] = batch_y.reshape((batch_y.shape[0])).cpu()
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            with torch.no_grad():
                mu = predictor(batch_x)         
                out_mu[current_index:current_index + mu.shape[0]] = mu.reshape((mu.shape[0])).cpu()

            current_index += mu.shape[0]

        out_mu = transforms["val"]["y"].backward(out_mu)
        batch_y_true = transforms["val"]["y"].backward(batch_y_true)
        out_mu_final  = out_mu.reshape(-1,1)

        return out_mu_final, batch_y_true

    def train_single_predictor(
            self,
            train_config : dict,
            train_dl : torch.utils.data.DataLoader,
            val_dl : torch.utils.data.DataLoader,
            device : torch.device,
            transforms : dict,
            logger : utils.logger.NetworkLogger,
            logger_prefix : str = "predictor",
            weighted_training = False,
            metrics : dict= None,
            categorize : bool = False,
            checkpoint = None
            ) -> None:
            
            num_iters = train_config.get('num_iter', 100)
            val_every = train_config.get('val_every', num_iters)
            print_every = train_config.get('print_every', num_iters + 1)
            train_type = train_config.get('train_type', 'iter')
            weight_type = train_config.get('weight_type', 'both')

            if checkpoint:
                save_time = time.strftime("%Y%m%d-%H%M%S")
                
            if train_type == "epoch":
                num_iters = int(num_iters * len(train_dl))
                val_every = int(val_every * len(train_dl))
                print_every = int(print_every * len(train_dl))

            self.device = device
            predictor = self.predictor.to(device)
            optimizer = self.predictor_optimizer
            iter_dl = iter(train_dl)
            #TODO : Run the dataloader once to calculate weights beforehand.
            
            for num_iter in tqdm.tqdm(range(num_iters), position=0, leave=True):
                # move data to device'
                try:
                    batch_x, batch_y = next(iter_dl)
                except:
                    iter_dl = iter(train_dl)
                    batch_x, batch_y = next(iter_dl)

                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()

                mu_train = predictor(batch_x)
                # calculate weight per sample
                if weighted_training:
                    weights = self._calculate_weights(batch_x,batch_y, weight_type = weight_type).detach().to(device)
                    loss = torch.mean(torch.abs(mu_train - batch_y) * weights)
                else:
                    loss = torch.mean(torch.abs(mu_train - batch_y))
                    
                if torch.isnan(loss):
                    raise ValueError('Loss is NaN')

                # calculate the loss and update the weights
                optimizer.zero_grad()
                loss.backward()

                # clip the gradients
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 10.0)

                optimizer.step()

                if train_type == 'epoch':
                    # if epoch is passed
                    if (num_iter + 1) % len(train_dl) == 0:
                        if self.predictor_scheduler is not None:
                            self.predictor_scheduler.step()
                else:
                    if self.predictor_scheduler is not None:
                        self.predictor_scheduler.step()

                mu_train = mu_train.detach()
                
                batch_y_transformed = transforms["train"]["y"].backward(batch_y)
                mu_train_transformed = transforms["train"]["y"].backward(mu_train)

                # log the criterion/loss, current lr
                logger.log_metric(
                    metric_dict = {
                    logger_prefix + "_train_loss": loss.item(),
                    logger_prefix + "_lr": optimizer.param_groups[0]['lr'],
                    },
                    step = (num_iter + 1),
                )

                train_metric_list = metrics.get("train", None)
                if train_metric_list is not None:
                    train_metric_list.forward(mu_train_transformed, batch_y_transformed)
                    logger.log_metric(
                        metric_dict = train_metric_list.get_metrics(add_prefix = logger_prefix),
                        step = (num_iter + 1),
                    )

                if (num_iter + 1) % val_every == 0:
                    val_mu, val_true = self.evaluate_predictor(val_dl, predictor, transforms, device)
                    val_metric_list = metrics.get("val", None)
                    if val_metric_list is not None:
                        if categorize:
                            assert hasattr(val_dl.dataset, "get_categories"), "Dataset class must have get_categories method"
                            categories = val_dl.dataset.get_categories()

                            # get category labels for all val_true
                            val_true_cat = np.array([categories.get(int(val_true[i]),"zero-shot") for i in range(len(val_true))])
                            for cat in np.unique(val_true_cat):
                                cat_mask = val_true_cat == cat
                                cat_val_true = val_true[cat_mask]
                                cat_val_mu = val_mu[cat_mask]

                                val_metric_list.forward(cat_val_mu, cat_val_true)
                                logger.log_metric(
                                    metric_dict = val_metric_list.get_metrics(add_prefix = logger_prefix + "_val_" + cat),
                                    step = (num_iter + 1),
                                )
                        val_metric_list.forward(val_mu, val_true)
                        logger.log_metric(
                            metric_dict = val_metric_list.get_metrics(add_prefix = logger_prefix + "_val"),
                            step = (num_iter + 1),
                        )
                    if checkpoint is not None:
                        checkpoint_folder = os.path.join(checkpoint, save_time)
                        if not os.path.exists(checkpoint_folder):
                            os.makedirs(checkpoint_folder)
                        torch.save(predictor.state_dict(), os.path.join(checkpoint_folder,f"predictor_network.pth"))
            return predictor
