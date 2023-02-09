# Uncertainty & Regression
## 1. Usage
### Configs
To specify the parameters and training strategies of the model, `yaml` file format
is being used. 


```yaml
network: # basic MLP layer configuration
  num_networks: 5
  layer_sizes: [64,128,256]
dataset: # for now, just use toy-dataset
  class: 'xls'
  xls_path: "Concrete_Data.xls"
train:
  batch_size : 32   
  num_iter : 10000
  print_every : 2500
test:
  batch_size : 32
logger:
  type: 'wandb'
  project: 'uncertainty-estimation'
  entity: 'kbora'
```
> Example config file for Concrete dataset. 
> 
## 2. File Structure
```
|-- ./estimations
|   |-- ./estimations/__init__.py
|   |-- ./estimations/ensemble.py
|-- ./tools
|   `-- ./tools/train.py
|-- ./utils
|   |-- ./utils/__init__.py
|   |-- ./utils/logger.py
|   |-- ./utils/device.py
|   |-- ./utils/clearml.py
|-- ./figures
|   |-- ./figures/non-weight.png
|   `-- ./figures/weighted.png
|-- ./datasets
|   |-- ./datasets/__init__.py
|   |-- ./datasets/toydata.py
|   |-- ./datasets/toyfunc.py
|   |-- ./datasets/concrete.py
|-- ./configs
|   |-- ./configs/concrete_dataset.yaml
|   `-- ./configs/toy_dataset.yaml                                                              
```
---
## Papers:
This section covers some of the well-known paper for imbalanced learning and uncertainity estimation for regression tasks.
### Imbalanced Regression:
Imbalanced regression are the collection of method that try to increase model performances for 
the areas where model in unsure due to the lack of training data.
- [Balanced MSE for Imbalanced Visual Regression](https://arxiv.org/abs/2203.16427) | [github]()
- [Density‑based weighting for imbalanced regression](https://link.springer.com/article/10.1007/s10994-021-06023-5) | []()
- [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554) | []()
### Uncertainity Estimation:
Uncertainity of the models and the data can be estimated with various methods which are usually
classified as (i) bayesian and (ii) non-bayesian methods.

#### (i) Bayesian Methods
- [Depth Uncertainty in Neural Networks](https://arxiv.org/abs/2006.08437) | [github](https://github.com/cambridge-mlg/DUN)
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) | [github](https://github.com/cambridge-mlg/DUN) 

#### (ii) Non-bayesian Methods
- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474)

