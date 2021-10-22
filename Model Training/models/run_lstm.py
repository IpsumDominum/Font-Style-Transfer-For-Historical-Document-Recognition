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


def train_lstm_model(LOAD,device=torch.device("cpu")):
    print("Initiating Dataset...")
    directory = os.path.join("segmented")
    filename = os.path.join("data","train_label.pkl")
    train_labels = pickle.load(open(filename, 'rb'))
    dataset = SegDataset(directory,train_labels,device)

    print("Initiating Dataloader...")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)

    print("Initiated")
    model = LSTMOCR(300,256,len_vocab).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if(LOAD!="0"):
        PATH = os.path.join("models","LSTM_ONLY",str(LOAD))
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        pass
    loss_function = nn.MSELoss()

    from tqdm import tqdm

    loss_history = []
    for epoch in range(0,10):
        avg_loss = 0
        batch_size = 1
        num = 0
        loss = torch.FloatTensor([0])
        for batch in tqdm(data_loader):
            inputs = batch[0][0]
            label = batch[1][0,1:-1]
            label_codes = batch[2][0,1:-1]
            if(num%batch_size==0):
                optimizer.zero_grad()
                model.zero_grad()
            out = model(inputs)
            #loss = loss_function(out,label_codes) * (1/batch_size)
            loss = loss_function(out,label) * (1/batch_size)
            avg_loss = (avg_loss+loss.item()*batch_size)/2
            loss.backward()

            if(num%batch_size==0):
                optimizer.step()
            num +=1
            if(num%100==0):
                print("Epoch : {} | AVG_LOSS : {} ".format(epoch,avg_loss))
                loss_history.append(avg_loss)
            if(num%5000==0):
                PATH = os.path.join("models","LSTM_ONLY",str(int(LOAD)+num))
                torch.save({
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'loss_history':loss_history
                    }, PATH)


def test_lstm_model(LOAD,device=torch.device("cpu")):
    print("Initiating Dataset...")
    directory = os.path.join("segmented_test")
    filename = os.path.join("data","test_seg_labels.pkl")
    test_labels = pickle.load(open(filename, 'rb'))
    dataset = SegDataset(directory,test_labels,device)

    print("Initiating Dataloader...")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)

    print("Initiated")
    model = LSTMOCR(300,256,len_vocab).to(device)
    
    PATH = os.path.join("models","LSTM_ONLY",str(LOAD))
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model"])

    loss_function =  nn.MSELoss()

    from tqdm import tqdm
    loss_history = []
    for epoch in range(0,10):
        avg_loss = 0
        for batch in tqdm(data_loader):
            inputs = batch[0][0]
            label = batch[1][:,1:-1]
            img = batch[3][0].detach().numpy()

            out = model(inputs)

            loss = loss_function(out,label[0])        
            avg_loss = (avg_loss+loss.item())/2

            print("===================")
            print(out_to_string(out))
            print(out_to_string(label[0]))
            print("===================")
            
            cv2.imshow('img',img)
            k = cv2.waitKey(0)
            if(k==ord('q')):
                cv2.destroyAllWindows()
                exit()

        

if __name__=="__main__":
    pass