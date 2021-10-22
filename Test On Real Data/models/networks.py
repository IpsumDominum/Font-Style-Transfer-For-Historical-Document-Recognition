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
from scripts.utils import out_to_string,string_to_embed,len_vocab,get_start_embed_one_hot
import torch
from torch import nn
import torchvision

#self.device = torch.self.device("cuda" if torch.cuda.is_available() else "cpu")


"""
ResNetEncoder and Attention from:
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""
class ResNetEncoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(ResNetEncoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-4]
        self.resnet = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()
    def forward(self, images):
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        #out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        #out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out
    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class ConvClassifier(nn.Module):
    def __init__(self,max_length):
        super(ConvClassifier, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.max_string_length = max_length
        self.resnet = nn.Sequential(*modules)
        self.bn = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(16384,len_vocab*self.max_string_length)

    def forward(self, images):
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.fc(self.bn(out).view(images.size(0),-1))
        output = F.log_softmax(out, dim=1)
        return output.view(images.size(0),self.max_string_length,len_vocab)

class Attention(nn.Module):
    def __init__(self,encode_dimension=128,decode_dimension=len_vocab,attention_dim=128):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encode_dimension, attention_dim) 
        self.decoder_att = nn.Linear(decode_dimension, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, encoder_out,decoder_out):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_out)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class LSTMDecoder(nn.Module):
    def __init__(self,device=torch.device("cpu")):
        super(LSTMDecoder, self).__init__()
        self.device = device
        self.encoder_dim = 128
        self.decoder_dim = 200
        self.attention_dim = 128
        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)  # attention network
        self.dropout = nn.Dropout(p=0.1)
        self.decode_step = nn.LSTMCell(len_vocab + self.encoder_dim, self.decoder_dim, bias=True)  # (input size, hidden size)
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial cell state of LSTMCell
        #self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        #self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, len_vocab)  # linear layer to find scores over vocabulary
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    def forward(self, encoder_out, caption_lengths=None,target_sequence=None,mode="Train",max_len=0,teacher_rate=0):
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)  # (batch_size, num_features, encoder_dim)
    
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        if(mode=="Train"):
            # Sort input data by decreasing lengths
            caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
            encoder_out = encoder_out[sort_ind]
            target_sequence = target_sequence[sort_ind]
            
            #Turn caption lengths to list for batch ordering
            caption_lengths = caption_lengths.tolist()

            predictions = torch.zeros(batch_size, max_len,len_vocab).to(self.device)
            alphas = torch.zeros(batch_size,  max_len, encoder_out.size(1)).to(self.device)
            for t in range(max(caption_lengths)):
                #Only choose the indices in the batch for which the sequence length exceeds
                #Current time step
                batch_size_t = sum([l > t for l in caption_lengths])                
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],h[:batch_size_t])
                #gate = self.sigmoid(self.f_beta(h[:batch_size]))  # gating scalar, (batch_size_t, encoder_dim)
                #attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = self.decode_step(
                    torch.cat([target_sequence[:batch_size_t,t,:], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                pred = self.softmax(self.fc(self.dropout(h)))
                """
                if(t != target_sequence.shape[1]-2):
                    #Use teacher
                    next_input = target_sequence[,t+1,:]
                else:
                    #Use prev output
                    next_input = pred
                """
                predictions[:batch_size_t,t,:] = pred
                alphas[:batch_size_t,t,:] = alpha
            return predictions,alphas
        else:
            predictions = []
            alphas = []
            next_input = get_start_embed_one_hot()
            t = 1
            while True:
                attention_weighted_encoding, alpha = self.attention(encoder_out,h)
                h, c = self.decode_step(
                    torch.cat([next_input, attention_weighted_encoding], dim=1),
                    (h, c))  # (batch_size_t, decoder_dim)
                pred = self.softmax(self.fc(self.dropout(h)))
                #if(t<target_sequence.shape[1]-1):
                #    next_input = target_sequence[:,t,:]
                #else:
                next_input = pred
                t +=1
                if(out_to_string(pred)=="<end>" or t >100):
                    break
                predictions.append(pred)
                alphas.append(alpha)
            predictions_torch = torch.zeros(batch_size,len(predictions), len_vocab).to(self.device)
            alphas_torch = torch.zeros(batch_size,len(alphas), encoder_out.shape[1]).to(self.device)
            for t in range(len(predictions)):
                predictions_torch[:,t,:] = predictions[t]
                alphas_torch[:,t,:] = alphas[t]
            #print(alphas_torch.shape)
            return predictions_torch,alphas_torch

class LSTMOCR(nn.Module):
    def __init__(self, input_dim,hidden_dim, vocab_size):
        super(LSTMOCR, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.hidden2out(lstm_out.view(len(inputs), -1))
        output = F.softmax(output, dim=1)
        return output

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        pass
    def forward(self,pred, target):
        """This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """

        smooth = 1.

        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        
        return  -(1-(2. * intersection + smooth) / (A_sum + B_sum + smooth))