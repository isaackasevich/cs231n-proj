import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch.autograd import Variable

import sys
sys.path.append('../dataprocessing')
from dataloader import BlurDataset

import torch.nn as nn
import torch.nn.functional as F

'''
Define GAN classes
'''
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, -1)  

class Generator(nn.Module):
    def __init__(self, img_shape, channel_1=20, channel_2=16, channel_3=8, channel_4=6):
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
    def __init__(self, img_shape, channel_1=16, channel_2=16, channel_3=8, hidden_size=200):
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
    
def bce_loss(input, target):
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
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def generator_loss(fake_scores):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    N = fake_scores.shape
    
    true_labels = torch.ones(N).type(dtype)
    
    loss = bce_loss(fake_scores, true_labels)
    
    return loss


def discriminator_loss(true_scores, fake_scores):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    N = true_scores.shape
    
    true_labels = torch.ones(N).type(dtype)
    fake_labels = torch.zeros(N).type(dtype)
    
    true_loss = bce_loss(true_scores, true_labels)
    fake_loss = bce_loss(fake_scores, fake_labels)
    
    loss = true_loss + fake_loss
    
    return loss

'''
Initialization 
'''
num_epochs = 1
batch_size = 100
channels = 3
size = 200 
img_shape = (channels, size, size)
lr = 2e-4
b1 = .5
b2 = .999
sample_interval = 25
save_interval = 25
validation_rate = 1   #Test on validation set once an epoch

input_size = img_shape[0] * img_shape[1] * img_shape[2]
   
if torch.cuda.is_available(): dtype = torch.cuda.FloatTensor
else: dtype = torch.FloatTensor
    
pixel_loss = torch.nn.MSELoss().type(dtype)

#Models
generator = Generator(img_shape).type(dtype)
discriminator = Discriminator(img_shape).type(dtype)

# Optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

'''
Load Data 
'''
    
data_path='../../data/coco'

data = BlurDataset.from_single_dataset(data_path)
train_dataloader = data.loader(batch_size=batch_size)
val_dataloader = data.loader(split='val', batch_size=batch_size)
'''
Train

TODO:
    -Experiment with params
'''
def save_losses(losses, path):
    with open(path, 'w') as f:
        for item in losses:
            f.write("{}\n".format(item))

def train_batches(epoch, generator, discriminator, dataloader, it=0, train=True, save=True):
    losses = []
    #Loop through batch
    for i, (imgs, tgts) in enumerate(dataloader):

        #Send imgs to gpu
        imgs = imgs.type(dtype)
        tgts = tgts.type(dtype)

        ### Train generator ###
        g_optimizer.zero_grad()

        # Generate images using the generator, find loss
        gen_imgs = generator(imgs)
        g_loss = generator_loss(discriminator(gen_imgs))
        if train:
            g_loss.backward()
            g_optimizer.step()

        ### Train discriminator ###
        d_optimizer.zero_grad()

        d_loss = discriminator_loss(discriminator(tgts), discriminator(gen_imgs.detach()))
        if train:
            d_loss.backward()
            d_optimizer.step()

        iteration = it + epoch * len(dataloader) + i
        pathval = "train" if train else "val"
        
        p_loss = pixel_loss(gen_imgs, tgts)
        
        print(pathval + " [Epoch %d/%d] [Batch %d/%d] [Iter %d] [G_Loss %f] [D_Loss %f] [P_Loss %f] "
        % (epoch, num_epochs, i, len(dataloader), iteration, g_loss, d_loss, p_loss))

        if iteration % sample_interval == 0:
            losses.append((g_loss.item(), d_loss.item(), iteration))
            save_losses(losses, "../../outputs/gan/" + pathval + "_losses.txt")
            save_image(imgs.data[:4], "../../outputs/gan/%d_" % iteration + pathval + "_input.png", nrow=2)
            save_image(gen_imgs.data[:4], "../../outputs/gan/%d_" % iteration + pathval + "_output.png", nrow=2)
            save_image(tgts.data[:4], "../../outputs/gan/%d_" % iteration + pathval + "_target.png", nrow=2)
        if save and (iteration % save_interval == 0): 
            torch.save(generator, "../../outputs/gan/generator.pt")
            torch.save(discriminator, "../../outputs/gan/discriminator.pt")

def train_model(generator, discriminator, save=True, it=0):
    
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_losses += train_batches(epoch, generator, discriminator, train_dataloader, it)

        if epoch % validation_rate == 0:
            val_losses += train_batches(epoch, generator, discriminator, val_dataloader, it, train=False, save=False)
            
            
            
            
 
  
            
            
            
            
            
            
        