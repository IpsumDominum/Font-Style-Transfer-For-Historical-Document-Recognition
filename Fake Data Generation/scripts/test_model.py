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

filename = "train_label.pkl"
train_labels = pickle.load(open(filename, 'rb'))
filename = "vocab.pkl"
vocab = pickle.load(open(filename, 'rb'))
filename = "reverse_vocab.pkl"
reverse_vocab = pickle.load(open(filename, 'rb'))

class LSTMOCR(nn.Module):

    def __init__(self, input_dim,hidden_dim, vocab_size):
        super(LSTMOCR, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.hidden2out(lstm_out.view(len(inputs), -1))
        output = F.log_softmax(output, dim=1)
        return output

def out_to_string(out):
    s = ""
    for i in out:
        s += vocab[torch.argmax(i).item()]
    return s
def string_to_embed(string):
    embed = torch.zeros((len(string),len(vocab)))
    #embed = torch.zeros((len(string))).type(torch.LongTensor)
    for idx,c in enumerate(string):
        #one hot encode the characters
        one_hot = torch.zeros((len(vocab)))
        one_hot[reverse_vocab[c]] = 1
        embed[idx] = one_hot
        #embed[idx] = reverse_vocab[c]
    return embed
data = []

PATH = os.path.join("models","LSTM_ONLY","29")
model = LSTMOCR(300,256,len(vocab))
model.load_state_dict(torch.load(PATH))
model.eval()

for item in os.listdir("segmented_lines"):
    img = cv2.imread(os.path.join("segmented_lines",item),0)
    text_length = len(train_labels[item])
    crop_size = img.shape[1]//text_length
    padded = np.pad(img,((0,0),(crop_size,crop_size)),"constant",constant_values=0)
    inputs = torch.zeros((text_length,1,300))
    for i in range(text_length):
        cropped = img[:,i*crop_size:(i+1)*crop_size]
        resized = cv2.resize(cropped,(30,10),interpolation=3)        
        inputs[i][0] = torch.from_numpy(resized.flatten()/255)
    embeded = string_to_embed(train_labels[item])

    out = model(inputs)
    print(out_to_string(embeded))
    print(out_to_string(out))
    cv2.imshow('img',img)
    k = cv2.waitKey(0)
    if(k==ord('q')):
        break
    print("=================")
    #data.append((img,inputs,embeded))




    
    

