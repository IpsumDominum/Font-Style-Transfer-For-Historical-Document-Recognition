import os
import pickle
from models.run_lstm import train_lstm_model
from models.run_attend_and_tell import train_attend_and_tell_model
from models.run_full_conv import train_full_conv_model
from models.run_U_net import train_style_transfer
from sys import argv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if __name__=="__main__":
    if(argv[1]=="attend"):
        train_attend_and_tell_model(argv[2],device)
    elif(argv[1]=="lstm"):
        train_lstm_model(argv[2])
    elif(argv[1]=="conv"):
        train_full_conv_model(argv[2])
    elif(argv[1]=="style"):
        train_style_transfer(argv[2],device)


