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
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from networks import UNet,DiceLoss
from scripts.utils import preprocess_style_transfer
class FontStyleTransfer:

    def __init__(self,LOAD,device=torch.device("cpu"),SAVE_DIR="STYLE"):
        print("Initiating model...")
        self.model = UNet(1,1).to(device)
        print("Model initiated")

        print("Loading checkpoint {}...".format(str(LOAD)))
        PATH = os.path.join("models",SAVE_DIR,str(LOAD))
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        print("LOADED : "+str(LOAD))
        self.device = device
    def test_on_image(self,inputs):
        inputs = preprocess_style_transfer(inputs)
        out = self.model(inputs)
        return out