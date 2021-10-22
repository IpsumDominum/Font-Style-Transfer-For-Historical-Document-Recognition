import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import os
import cv2
import numpy as np
from scripts.utils import string_to_embed,prep_image,string_to_embed_one_hot,pad_string,get_max_length
from scripts.binarize import binarize

class SegDataset(torch.utils.data.Dataset):
    """Some Information about SegDataset"""
    def __init__(self,directory,train_labels,device=torch.device("cpu")):
        super(SegDataset, self).__init__()
        print("initating dataset")
        self.all_items = os.listdir(directory)
        self.root_dir = directory
        self.train_labels = train_labels
        self.device = device
    def __getitem__(self, index):
        item = self.all_items[index]
        img = cv2.imread(os.path.join(self.root_dir,item),0)
        text_length = len(self.train_labels[item])
        crop_size = img.shape[1]//text_length
        padded = np.pad(img,((0,0),(crop_size,crop_size)),"constant",constant_values=255)
        inputs = torch.zeros((text_length,1,300))
        inputs_raw = 255 - padded
        for i in range(text_length):
            cropped = img[:,i*crop_size:(i+1)*crop_size]
            resized = cv2.resize(cropped,(30,10),interpolation=3)        
            inputs[i][0] = torch.from_numpy(resized.flatten()/255)
        
        label_string = self.train_labels[item]
        embeded = string_to_embed_one_hot(label_string)
        embeded_code = string_to_embed(label_string)
        return (inputs.to(self.device),embeded.to(self.device),embeded_code.to(self.device),inputs_raw)
    def __len__(self):
        return len(self.all_items)


class LineDataset(torch.utils.data.Dataset):
    """Some Information about SegDataset"""
    def __init__(self,directory,labels,PAD=True,device=torch.device("cpu")):
        super(LineDataset, self).__init__()
        print("initating line dataset")
        self.all_items = os.listdir(directory)
        print("Directory loaded")
        self.root_dir = directory
        self.labels = labels
        self.PAD = PAD
        if(self.PAD):
            self.max_len = get_max_length(labels)
        else:
            self.max_len = 0
        self.device = device
        print("Dataset initiated")
    def get_max_len(self):
        return self.max_len
    def __getitem__(self, index):
        item = self.all_items[index]
        img = cv2.imread(os.path.join(self.root_dir,item))
        img = prep_image(img)
        label_string = self.labels[item]
        embeded_length = len(label_string)+1
        if(self.PAD):
            label_string = pad_string(label_string,self.max_len)
        embeded = string_to_embed_one_hot(label_string)
        embeded_code = string_to_embed(label_string)[1:]
        return (img.to(self.device),embeded.to(self.device),embeded_code.to(self.device),embeded_length.to(self.device))
    def __len__(self):
        return len(self.all_items)

class StyleTransferDataset(torch.utils.data.Dataset):
    """Some Information about SegDataset"""
    def __init__(self,directory,device=torch.device("cpu")):
        super(StyleTransferDataset, self).__init__()
        self.sample_dir = os.path.join(directory,"fonts")
        self.ground_truth_dir = os.path.join(directory,"ground_truth")
        self.all_items = os.listdir(self.sample_dir)
        self.device = device
    def __getitem__(self, index):
        item = self.all_items[index]
        img = cv2.imread(os.path.join(self.sample_dir,item),0)
        #img = binarize(img,mode="sauvola")
        img = prep_image(img,gray=True)
        ground_truth = cv2.imread(os.path.join(self.ground_truth_dir,item),0)
        ground_truth = prep_image(ground_truth,gray=True)
        return (img.to(self.device),ground_truth.to(self.device))
    def __len__(self):
        return len(self.all_items)





"""
data_loader = []
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
    data_loader.append((inputs,embeded))
"""