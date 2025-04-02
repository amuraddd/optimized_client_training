import os
import torch
import math
import pandas as pd
from collections import defaultdict
from config_folder import client_config_file

def get_parameters(net):
    """
    Get a list of numpy arrays as model parameters
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_torch_device(manual_seed=42, strategy="FedAvg"):
    """
    Set device for mps, cuda
    Set manual seed
    """
    if torch.backends.mps.is_available():
        device = client_config_file.DEVICE_MPS
        torch.mps.manual_seed(manual_seed)
    elif torch.cuda.is_available():
        device_count = [i for i in range(torch.cuda.device_count())]
        if strategy=="FedAvg":#expand to all strategies and use multiple processes to parallelize
            device = f"cuda:{device_count[0]}"
        elif strategy=="FedAvgM":#expand to all strategies and use multiple processes to parallelize
            device = f"cuda:{device_count[1]}"
        elif strategy=="FedProx":#expand to all strategies and use multiple processes to parallelize
            device = f"cuda:{device_count[2]}"
        else:#expand to all strategies and use multiple processes to parallelize
            device = f"cuda:{device_count[0]}"
        device = 'cuda'
        torch.cuda.manual_seed(manual_seed)
    else:
        device = client_config_file.DEVICE_CPU
        torch.manual_seed(manual_seed)

    print(f"Device: {device} | Manual Seed set to {manual_seed}")
    return device

def get_class_distribution(y):
    """
    Given a list or array of labels, get their class distribution.
    """
    weights = pd.DataFrame(y)[0].value_counts(sort=False)/pd.DataFrame(y).value_counts().sum()
    weights = weights.round(2)
    return list(weights.to_dict().values())

def notebook_line_magic():
    """
    Avoid having to restart kernel when working with python scripts
    """
    from IPython import get_ipython
    ip = get_ipython()
    ip.run_line_magic("reload_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    print("Line Magic Set")
    
def calc_logprob(mu_v, var_v, actions_v):
    """
    log probability of the gaussian distribution
    """
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")