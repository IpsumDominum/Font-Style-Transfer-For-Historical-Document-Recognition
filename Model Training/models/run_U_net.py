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
import pickle
from scripts.utils import out_to_string,string_to_embed
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dataset import StyleTransferDataset
from networks import UNet,DiceLoss


def train_style_transfer(LOAD=0,device=torch.device("cpu")):
    print("Initiating Dataset...")
    directory = os.path.join("Style_Transfer_Data","Train")
    dataset = StyleTransferDataset(directory,device)

    print("Initiating Dataloader...")
    BATCH_SIZE = 5
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=False)

    print("Initiating models...")
    model = UNet(1,1).to(device)

    print("Initiating optimizers")
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Models and optimizers initiated")

    SAVE_DIR = "STYLENEW"

    if(LOAD!="0"):
        print("Loading checkpoints...")
        PATH = os.path.join("models",SAVE_DIR,str(LOAD))
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_history = checkpoint["loss_history"]
        print("LOADED : "+str(LOAD))
    else:
        loss_history = []
        pass
    
    loss_function = nn.MSELoss() #nn.CrossEntropyLoss()
    from tqdm import tqdm
    #loss_history = []
    SAVE_GAP = 1000
    num = 0
    running_loss = np.zeros((SAVE_GAP))
    for epoch in range(0,10):
        for batch in tqdm(data_loader):
            inputs = batch[0]
            ground_truth = batch[1]

            optimizer.zero_grad()
            out = model(inputs)
            loss = loss_function(out[:,0,:,:],ground_truth[:,0,:,:])
            #loss = loss_function(out,label_codes) #Cross Entropy
            running_loss[num%SAVE_GAP] = loss.item()
            loss.backward()
            optimizer.step()
            num += BATCH_SIZE
            #TEACHER_RATE = max(TEACHER_RATE-0.00001,0)
            if(num%100==0):
                print("Epoch : {} | AVG_LOSS : {} ".format(epoch,running_loss))
            if(num%SAVE_GAP==0 and num!=0):
                loss_history.append(np.array(running_loss).mean())
                PATH = os.path.join("models",SAVE_DIR,str(int(LOAD)+num))
                torch.save({
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'loss_history':loss_history
                    }, PATH)

def test_style_transfer(LOAD,device=torch.device("cpu")):
    print("Initiating Dataset...")
    directory = os.path.join("Style_Transfer_Data","Test")
    dataset = StyleTransferDataset(directory,device)

    print("Initiating Dataloader...")
    BATCH_SIZE = 1
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=False)

    print("Initiating model...")
    model = UNet(1,1).to(device)

    model.eval()

    print("Model initiated")

    SAVE_DIR = "STYLE"

    if(LOAD!="0"):
        print("Loading checkpoints...")
        PATH = os.path.join("models",SAVE_DIR,str(LOAD))
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model"])
        loss_history = checkpoint["loss_history"]
        print("LOADED : "+str(LOAD))
    else:
        loss_history = []
        pass
    loss_function = nn.MSELoss() #nn.CrossEntropyLoss()
    from tqdm import tqdm
    #loss_history = []
    for batch in tqdm(data_loader):
        inputs = batch[0]
        ground_truth = batch[1]

        out = model(inputs)

        img = np.swapaxes(ground_truth[0].cpu().detach().numpy(),0,2)
        img = np.swapaxes(img,0,1)

        out_img = np.swapaxes(out[0].cpu().detach().numpy(),0,2)
        out_img = np.swapaxes(out_img,0,1)

        input_img = np.swapaxes(inputs[0].cpu().detach().numpy(),0,2)
        input_img = np.swapaxes(input_img,0,1)
        print("===================")
        cv2.imshow('img',img)
        cv2.imshow('input',input_img)
        cv2.imshow('out',out_img)
        k = cv2.waitKey(0)
        if(k==ord('q')):
            break
    cv2.destroyAllWindows()