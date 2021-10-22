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
import sys
sys.path.append("scripts")
from utils import out_to_string,string_to_embed,len_vocab
from models.run_attend_and_tell import test_attend_and_tell
from models.run_lstm import test_lstm_model
from models.run_U_net import test_style_transfer
from sys import argv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
if __name__=="__main__":
    if(argv[1]=="attend"):
        test_attend_and_tell(argv[2],device)
    elif(argv[1]=="lstm"):
        test_lstm_model(argv[2])
    elif(argv[1]=="conv"):
        pass
        #test_full_conv_model(argv[2])
    elif(argv[1]=="style"):
        test_style_transfer(argv[2],device)