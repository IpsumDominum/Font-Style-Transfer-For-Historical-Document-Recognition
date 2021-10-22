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
from dataset import LineDataset
from networks import ResNetEncoder,LSTMDecoder



def train_attend_and_tell_model(LOAD=0,device=torch.device("cpu")):
    print("Initiating Dataset...")
    directory = os.path.join("segmented")
    filename = os.path.join("data","train_label.pkl")
    train_labels = pickle.load(open(filename, 'rb'))
    dataset = LineDataset(directory,train_labels,device)

    print("Initiating Dataloader...")
    BATCH_SIZE = 20
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=False)

    print("Initiating models...")
    encoder = ResNetEncoder().to(device)
    #encoder.fine_tune(True)
    decoder = LSTMDecoder(device=device).to(device)

    print("Initiating optimizers")
    encode_optimizer = optim.Adam(encoder.parameters(), lr=0.01)
    decode_optimizer = optim.Adam(decoder.parameters(),lr=0.01)

    print("Models and optimizers initiated")

    SAVE_DIR = "ATTEND3"
    
    if(LOAD!="0"):
        print("Loading checkpoints...")
        PATH = os.path.join("models",SAVE_DIR,str(LOAD))
        checkpoint = torch.load(PATH)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encode_optimizer.load_state_dict(checkpoint["optimizer_encode"])
        decode_optimizer.load_state_dict(checkpoint["optimizer_decode"])
        loss_history = checkpoint["loss_history"]
        print("LOADED : "+str(LOAD))
    else:
        loss_history = []
        pass
    loss_function = nn.MSELoss() #nn.CrossEntropyLoss()
    from tqdm import tqdm
    #loss_history = []
    MAX_LEN = dataset.get_max_len()+1
    for epoch in range(0,10):
        avg_loss = 0
        TEACHER_RATE = 1
        num = 0
        for batch in tqdm(data_loader):
            inputs = batch[0]
            label = batch[1]
            label_codes = batch[2]
            label_lengths = batch[3]

            encode_optimizer.zero_grad()
            decode_optimizer.zero_grad()
            encode = encoder(inputs)
            out,alphas = decoder(encode,caption_lengths=label_lengths,target_sequence=label,mode="Train",max_len=MAX_LEN,teacher_rate=TEACHER_RATE)


            loss = loss_function(out[0],label[0][1:]) #MSE
            #loss = loss_function(out,label_codes) #Cross Entropy
            avg_loss = (avg_loss+loss.item())/2
            loss.backward()
            encode_optimizer.step()
            decode_optimizer.step()
            num += BATCH_SIZE
            #TEACHER_RATE = max(TEACHER_RATE-0.00001,0)
            if(num%100==0):
                print("Epoch : {} | AVG_LOSS : {} | TEACHER_RATE : {}".format(epoch,avg_loss,TEACHER_RATE))
        loss_history.append(avg_loss)
        PATH = os.path.join("models",SAVE_DIR,str(int(LOAD)+epoch+1))
        torch.save({
            'encoder':encoder.state_dict(),
            'decoder':decoder.state_dict(),
            'optimizer_encode':encode_optimizer.state_dict(),
            'optimizer_decode':decode_optimizer.state_dict(),
            'loss_history':loss_history
            }, PATH)

def test_attend_and_tell(checkpoint_num,device=torch.device("cpu")):
    print("Initiating Dataset...")
    directory = os.path.join("segmented")
    filename = os.path.join("data","train_label.pkl")
    train_labels = pickle.load(open(filename, 'rb'))
    dataset = LineDataset(directory,train_labels,device)

    print("Initiating Dataloader...")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)

    SAVE_DIR = "ATTEND3"

    encoder = ResNetEncoder().to(device)
    decoder = LSTMDecoder().to(device)
    
    PATH = os.path.join("models",SAVE_DIR,str(checkpoint_num))
    checkpoint = torch.load(PATH)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()
    
    loss_function =  nn.CrossEntropyLoss() #nn.MSELoss()
    from tqdm import tqdm
    for batch in data_loader:
        inputs = batch[0]
        label = batch[1]
        label_codes = batch[2]
  
        encode = encoder(inputs)
        out,alphas = decoder(encode,label,mode="Test")

        #loss = loss_function(out[0],label_codes[0])
        #print(loss)

        img = np.swapaxes(inputs[0].cpu().detach().numpy(),0,2)
        img = np.swapaxes(img,0,1)

        print("===================")
        print(out.shape)
        print(out_to_string(out[0]))
        print(out_to_string(label_codes[0],mode="codes"))
        print("===================")
        cv2.imshow('img',img)
        k = cv2.waitKey(0)
        if(k==ord('q')):
            break

     
