import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Flatten(nn.Module):
    """
    nn module from class assignment that flattens image
    """
    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, -1)  
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    """
    weight initialization scheme
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        
    
class ResBlock(nn.Module):
    """
    residual block. conv -> batchnorm -> leakyRelu -> conv -> batchnorm
    """
    def __init__(self, num_filters, k=5, pad=2, s=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 
                               kernel_size=k, 
                               padding=pad,
                               stride=s)
        self.bn1 = nn.BatchNorm2d(num_filters, 0.8)
        self.leaky = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 
                               kernel_size=k, 
                               padding=pad,
                               stride=s)
        self.bn2 = nn.BatchNorm2d(num_filters, 0.8)
       
        initialize_weights(self.conv1)
        initialize_weights(self.conv2)
        
        
    def forward(self, x):
        conv1_out = self.leaky(self.bn1(self.conv1(x)))
        conv2_out = self.bn2(self.conv2(conv1_out))
        return x + conv2_out
    
class WGANGenerator(nn.Module):
    def __init__(self, img_shape, noise_dim=100):
        super(WGANGenerator, self).__init__()
        in_channel, H, W = img_shape
        
        self.noise_proj = nn.Linear(in_features=noise_dim, 
                                  out_features=in_channel*H*W)
        initialize_weights(self.noise_proj)
        
        self.leaky = nn.LeakyReLU()
        self.unflat = Unflatten(-1, in_channel, H, W)
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=5, padding=2)
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.conv2 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        
    def forward(self, x_img, x_noise):
        noise = F.relu(self.noise_proj(x_noise))
        unflat_noise = self.unflat(noise)
        
        img_noise = x_img + unflat_noise
        out1 = self.leaky(self.conv1(img_noise))
        out2 = self.leaky(self.res1(out1))
        out3 = self.leaky(self.res2(out2))
        out = self.conv2(out3)
        return out
        
        
class WGANDiscriminator(nn.Module):
    def __init__(self, img_shape, channel_1=32, channel_2=32, channel_3=16, hidden_size=50):
        super().__init__()

        in_channel, H, W = img_shape
        
        self.conv1 = nn.Conv2d(in_channel, channel_1, 5, padding=2)        
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        self.conv3 = nn.Conv2d(channel_2, channel_3, 3, padding=1)
        
        fc_in = int(channel_3 * H * W / 64)
        self.fc1 = nn.Linear(fc_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        self.leakyRelu = nn.LeakyReLU(.01)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()
        
        initialize_weights(self.conv1)
        initialize_weights(self.conv2)
        initialize_weights(self.conv3)
        initialize_weights(self.fc1)
        initialize_weights(self.fc2)
        
    def forward(self, x):
        x = self.maxPool(self.leakyRelu(self.conv1(x)))
        x = self.maxPool(self.leakyRelu(self.conv2(x)))
        x = self.maxPool(self.leakyRelu(self.conv3(x)))

        x = self.flatten(x)
        x = self.leakyRelu(self.fc1(x))
        scores = self.fc2(x)

        return scores
    
def generator_loss(fake_scores):
    """
    Wasserstein gan loss function for generator
    """
    return -torch.mean(fake_scores)

def discriminator_loss(real_scores, fake_scores):
    """
    Wasserstein gan loss function for discriminator
    """
    return -(torch.mean(real_scores) - torch.mean(fake_scores))
            
        
        