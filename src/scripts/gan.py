import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg16
import math

'''
Initialization 
'''
num_epochs = 2
batch_size = 50
channels = 3
size = 200 
img_shape = (channels, size, size)
lr = 1e-3
b1 = .5
b2 = .999
sample_interval = 25
save_interval = 25
validation_rate = 1   #Test on validation set once an epoch

input_size = img_shape[0] * img_shape[1] * img_shape[2]
   
if torch.cuda.is_available(): dtype = torch.cuda.FloatTensor
else: dtype = torch.FloatTensor
    
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:3])
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.feature_extractor(img)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, -1) 
    
class Generator(nn.Module):
    def __init__(self, img_shape, channel_1=24, channel_2=18, channel_3=12, channel_4=8):
        super().__init__()
        
        in_channel, H, W = img_shape
        
        self.conv1 = nn.Conv2d(in_channel, channel_1, 3, padding=1)    
        self.batchNorm1 = nn.BatchNorm2d(channel_1)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(channel_2)
        self.conv3 = nn.Conv2d(channel_2, channel_3, 3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(channel_3)
        self.conv4 = nn.Conv2d(channel_3, channel_4, 3, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(channel_4)
        self.conv5 = nn.Conv2d(channel_4, in_channel, 3, padding=1)
        
        self.relu = nn.ReLU()

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        
    def forward(self, x):
        x = self.batchNorm1(self.relu(self.conv1(x)))
        x = self.batchNorm2(self.relu(self.conv2(x)))
        x = self.batchNorm3(self.relu(self.conv3(x)))
        x = self.batchNorm4(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        
        return x
        
        
class Discriminator(nn.Module):
    def __init__(self, img_shape, channel_1=16, channel_2=16, channel_3=8, hidden_size=100):
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
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        
    def forward(self, x):
        x = self.maxPool(self.leakyRelu(self.conv1(x)))
        x = self.maxPool(self.leakyRelu(self.conv2(x)))
        x = self.maxPool(self.leakyRelu(self.conv3(x)))
        
        x = self.flatten(x)
        x = self.leakyRelu(self.fc1(x))
        scores = self.fc2(x)

        return scores

def bce_loss(inp, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - inp.abs()
    loss = inp.clamp(min=0) - inp * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def generator_loss(fake_scores, feature_extractor, gen_imgs, tgt_imgs):
    N = fake_scores.shape
    
    true_labels = torch.ones(N).type(dtype)
    disc_loss = bce_loss(fake_scores, true_labels)
    
    content_loss = F.mse_loss(feature_extractor(gen_imgs), feature_extractor(tgt_imgs))
    
    loss = disc_loss*1e-3 + content_loss
    
    return loss
    
def discriminator_loss(true_scores, fake_scores):
    N = true_scores.shape
    
    true_labels = torch.ones(N).type(dtype)
    fake_labels = torch.zeros(N).type(dtype)
    
    true_loss = bce_loss(true_scores, true_labels)
    fake_loss = bce_loss(fake_scores, fake_labels)
    
    loss = true_loss + fake_loss
    
    return loss