import torch.nn.functional as F
import numpy as np

"""
file with utility functions to quantitatively evaluate images
also plotting functions go in here to plot losses and such
"""

def reconstruction_loss(outputs, targets):
    return F.mse_loss(outputs, targets)
    

def plot_losses(file_path):
    pass


def append_to_file(losses, file_path):
    with open(file_path, "a+") as f:
        f.write(losses)