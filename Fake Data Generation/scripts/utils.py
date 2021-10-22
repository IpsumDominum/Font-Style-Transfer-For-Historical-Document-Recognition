import torchvision.transforms as transforms
import torch
import pickle
import os
import cv2
import numpy as np
#filename = os.path.join("data","train_label.pkl")
#train_labels = pickle.load(open(filename, 'rb'))
filename = os.path.join("data","vocab.pkl")
vocab = pickle.load(open(filename, 'rb'))
filename = os.path.join("data","reverse_vocab.pkl")
reverse_vocab = pickle.load(open(filename, 'rb'))
len_vocab = len(vocab)


def out_to_string(out,mode="one_hot"):
    s = ""
    for i in out:
        if(mode=="one_hot"):
            s += vocab[torch.argmax(i).item()]
        else:
            s += vocab[i.item()]
    return s

def prep_image(img,normalize=False,gray=False):
    #img = cv2.resize(img, (256, 256))
    width_diff = 512 - img.shape[1]
    height_diff = 80 - img.shape[0]
    
    if(width_diff>0):
        if(gray==True):
            img = np.pad(img,((height_diff - height_diff//2,height_diff//2),(0,width_diff)),"constant",constant_values=255)
        else:
            img = np.pad(img,((height_diff - height_diff//2,height_diff//2),(0,width_diff),(0,0)),"constant",constant_values=255)
    img = cv2.resize(img, (512, 80))
    """
    cv2.imshow('hi',img)
    k = cv2.waitKey(0)
    if(k==ord('q')):
        cv2.destroyAllWindows()
        exit()
    """
    if(gray==True):
        img = np.array([img])
    else:
        img = img.transpose(2, 0, 1)

    img = 255 - img
    img = img / 255.
    
    img_show = np.swapaxes(img,0,2)
    img_show = np.swapaxes(img_show,0,1)

    """
    cv2.imshow('hi',img_show)
    k = cv2.waitKey(0)
    if(k==ord('q')):
        cv2.destroyAllWindows()
        exit()
    """ 
    img = torch.FloatTensor(img)
    if(normalize==True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)  # (3, 256, 256)
    else:
        image = img
    return image


def string_to_embed(string):
    string = ["<start>"]+[c for c in string] + ["<end>"]
    embed = torch.zeros((len(string))).type(torch.LongTensor)
    for idx,c in enumerate(string):
        embed[idx] = reverse_vocab[c]
    return embed

def get_start_embed_one_hot():
    string = ["<start>"]
    embed = torch.zeros((len(string),len(vocab)))
    for idx,c in enumerate(string):
        #one hot encode the characters
        one_hot = torch.zeros((len(vocab)))
        one_hot[reverse_vocab[c]] = 1
        embed[idx] = one_hot
    return embed

def string_to_embed_one_hot(string):
    string = ["<start>"]+[c for c in string] + ["<end>"]
    embed = torch.zeros((len(string),len(vocab)))
    for idx,c in enumerate(string):
        #one hot encode the characters
        one_hot = torch.zeros((len(vocab)))
        one_hot[reverse_vocab[c]] = 1
        embed[idx] = one_hot
    return embed

def get_max_length(labels):
    max_len = 0
    for label in labels:
        length = len(labels[label])
        if(length>max_len):
            max_len = length
    return max_len

def pad_string(string,max_len):
    if(len(string)<max_len):
        return string + (max_len-len(string))*" "
    else:
        return string
