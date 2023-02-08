import torch
import numpy as np

def set_device(preference = None):
    """
    Set device to available GPU, Metal or CPU (in order)
    Args:
        preference (str): 'cuda', 'mps' or 'cpu'

    Returns:
        device (torch.device): device to use
    """
    if not preference:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        if preference == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise ValueError('CUDA is not available')
        elif preference == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                raise ValueError('Metal is not available')
        elif preference == 'cpu':
            device = torch.device('cpu')
        else:
            raise ValueError('Invalid preference')
    return device
    
def set_random(seed = 42):
    """
    Set all the pseudo-random number generators to a fixed seed.
    Args:
        seed (int): seed to use
    
    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    