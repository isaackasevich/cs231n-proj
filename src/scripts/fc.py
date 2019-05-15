import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

img_shape = (3, 32, 32)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Expand(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), img_shape[0], img_shape[1], img_shape[2])
        

input_size = img_shape[0] * img_shape[1] * img_shape[2]
        
fc_net = nn.Sequential(
    Flatten(),
    nn.Linear(input_size, input_size),
    Expand(),
)

