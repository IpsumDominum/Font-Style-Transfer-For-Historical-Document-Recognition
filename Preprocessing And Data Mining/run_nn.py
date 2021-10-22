import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import random
from torch.autograd import Variable
import os
import cv2
import numpy as np
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=128*128):
        return input.view(input.size(0), size, 1, 1)
        
class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=128*128, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar
from tqdm import tqdm 
class HDRDataset(torch.utils.data.Dataset):
    """Some Information about HDRDataset"""
    def __init__(self,dataset_dirs=[]):
        super(HDRDataset, self).__init__()
        self.total_dirs = []
        for directory in tqdm(dataset_dirs):
            print("loading from "+directory)
            for item in os.listdir(directory):
                self.total_dirs.append(os.path.join(directory,item))
    def __getitem__(self, index):
        img_path = self.total_dirs[index]
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img,(64,64))
        img = np.array([img])
        return torch.from_numpy(img).type(torch.FloatTensor)/255
    def __len__(self):
        return len(self.total_dirs)

dataset = HDRDataset(dataset_dirs=[os.path.join("temp","crop")])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)
vae = VAE().type(torch.FloatTensor)

#break
def show_output(output_img,id="hi",wait=1):    
    cv2.imshow(id,cv2.resize(np.swapaxes(output_img.detach().cpu().numpy(),0,2),(512,512)))        
def cv_wait(wait=0):
    k = cv2.waitKey(wait)
    if(k==ord('q')):
        return True
    return False
lr = 0.01
optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay = 0.0001)

BCE_loss = nn.BCELoss(reduction = "sum")
MSE_loss = nn.MSELoss()
def loss(X, X_hat, mean, logvar):
    reconstruction_loss = MSE_loss(X_hat, X)
    #reconstruction_loss = BCE_loss(X_hat[0], X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

VISUALIZE = True
num_epochs = 20
for epoch in range(num_epochs):
    for idx,d in tqdm(enumerate(dataloader)):
        out,mu,logvar = vae.forward(d)
        l = loss(out, d.unsqueeze(0), mu, logvar) * 10
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if(VISUALIZE):
            show_output(d[0],id="hi2")
            show_output(out[0])
            q = cv_wait(1)
            if(q==True):break
        