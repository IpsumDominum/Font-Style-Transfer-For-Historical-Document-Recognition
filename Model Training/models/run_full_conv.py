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
from scripts.utils import out_to_string,string_to_embed,len_vocab,get_max_length
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dataset import LineDataset
from networks import ConvClassifier


def train_full_conv_model(LOAD):
    print("Initiating Dataset...")
    directory = os.path.join("segmented")
    filename = os.path.join("data","train_label.pkl")
    train_labels = pickle.load(open(filename, 'rb'))
    dataset = LineDataset(directory,train_labels,True)

    print("Initiating Dataloader...")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)

    print("Initiated")
    model = ConvClassifier(get_max_length(train_labels))
    SAVE_DIR = "CONV_FULL"
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if(LOAD!=0):
        PATH = os.path.join("models",SAVE_DIR,str(LOAD))
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        pass
    loss_function =  nn.MSELoss()

    from tqdm import tqdm

    loss_history = []
    for epoch in range(0,10):
        avg_loss = 0
        num = 0
        for batch in tqdm(data_loader):
            inputs = batch[0]
            label = batch[1][:,1:-1,:]
            label_codes = batch[1][:-1]
            model.zero_grad()
            out = model(inputs)
            #loss = loss_function(out[0],label[0])       
            loss = loss_function(out[0],label_codes[0])         
            avg_loss = (avg_loss+loss.item())/2
            loss.backward()
            optimizer.step()
            num +=1
            if(num %100==0):
                print("Epoch : {} | AVG_LOSS : {} ".format(epoch,avg_loss))
                loss_history.append(avg_loss)
            if(num%1000==0):
                PATH = os.path.join("models",SAVE_DIR,str(LOAD+num))
                torch.save({
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'loss_history':loss_history
                    }, PATH)


if __name__=="__main__":
    pass