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
from scripts.utils import out_to_string,string_to_embed,len_vocab
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dataset import SegDataset
from networks import LSTMOCR


def train_full_conv_model():
    print("Initiating Dataset...")
    directory = os.path.join("segmented")
    filename = os.path.join("data","train_label.pkl")
    train_labels = pickle.load(open(filename, 'rb'))
    dataset = SegDataset(directory,train_labels)

    print("Initiating Dataloader...")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)

    print("Initiated")
    LOAD = 0
    model = LSTMOCR(300,256,len_vocab)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if(LOAD!=0):
        PATH = os.path.join("models","LSTM_ONLY",str(LOAD-1))
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        pass
    loss_function =  nn.NLLLoss() #nn.MSELoss()

    from tqdm import tqdm

    loss_history = []
    for epoch in range(0,10):
        avg_loss = 0
        for batch in tqdm(data_loader):
            inputs = batch[0][0]
            label = batch[1][0]
            model.zero_grad()
            out = model(inputs)
            print(out.shape)
            print(label.shape)
            loss = loss_function(out,label)        
            avg_loss = (avg_loss+loss.item())/2
            loss.backward()
            optimizer.step()
        print("Epoch : {} | AVG_LOSS : {} ".format(epoch,avg_loss))
        loss_history.append(avg_loss)
        PATH = os.path.join("models","LSTM_ONLY",str(epoch))
        torch.save({
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'loss_history':loss_history
            }, PATH)


if __name__=="__main__":
    pass