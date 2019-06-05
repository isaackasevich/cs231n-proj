import matplotlib
import numpy as np
import matplotlib.pyplot as plt

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

if torch.cuda.is_available(): dtype = torch.cuda.FloatTensor
else: dtype = torch.FloatTensor

def plot_losses(path, title, save_path, batch_size=100):
    with open(path) as f:
        losses = f.readlines()
        losses = [line.strip()[1:-1] for line in losses]
        losses = np.array([line.split(',')[:2] for line in losses])

        iters = [int(i) * batch_size for i in losses[:,0]]
        losses = [float(i) for i in losses[:,1]]
        
        plt.plot(iters, losses, linewidth=2.0)
        plt.title(title)
        plt.xlabel('Images Seen')
        plt.ylabel('Reconstruction Error')
        plt.savefig(save_path)

def save_losses(losses, path):
    with open(path, 'a+') as f:
        for item in losses:
            f.write("{}\n".format(item))    

def generate_imgs(model_path, save_path, split='test'):
    
    data = BlurDataset.from_single_dataset('../../data/coco')
    dataloader = data.loader(split=split, batch_size=25)
                  
    model = torch.load(model_path)
    model.eval()
    
    
    for i, (imgs, tgts) in enumerate(dataloader):

        #Send imgs to gpu
        imgs = imgs.type(dtype)
        tgts = tgts.type(dtype)

        gen_imgs = model(imgs)
        
        recon_loss = F.mse_loss(gen_imgs, tgts)
        save_losses([recon_loss], save_path+"test_loss.txt")
        
        save_image(imgs.data[:25], save_path + split +"_input.png", nrow=5)
        save_image(gen_imgs.data[:25], save_path +split +"_output.png", nrow=5)
        save_image(tgts.data[:25], save_path +split +"_target.png", nrow=5)
        
        break

                  
                  
          
