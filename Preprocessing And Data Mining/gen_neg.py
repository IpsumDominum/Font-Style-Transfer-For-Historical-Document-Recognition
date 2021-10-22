import os
import numpy as np
import cv2
import sys
sys.path.append("scripts")
from binarize import binarize

save_data_dir = os.path.join("Extraction_Data","Neg")
pos_data_dir = os.path.join("Extraction_Data","Pos")

def perlin(x,y,seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u) # FIX1: I was using n10 instead of n01
    return lerp(x1,x2,v) # FIX2: I also had to reverse x1 and x2 here

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

kernel = np.ones((3,3),np.uint8)
from tqdm import tqdm
for idx,pos_item in tqdm(enumerate(os.listdir(pos_data_dir))):
    read = cv2.imread(os.path.join(pos_data_dir,pos_item))
    dilated = cv2.dilate(read,kernel,iterations=2)
    linx = np.linspace(0,5,read.shape[1],endpoint=False)
    liny = np.linspace(0,5,read.shape[0],endpoint=False)
    x,y = np.meshgrid(linx,liny) # FIX3: I thought I had to invert x and y here but it was a mistake
    noise = perlin(x,y,seed=2)
    noise_3 = np.zeros(read.shape)
    noise_3[:,:,0] = noise
    noise_3[:,:,1] = noise
    noise_3[:,:,2] = noise
    noise_3 += np.random.normal(read.shape)
    mixed = (dilated+abs(noise_3*255))%255
    final = mixed
    #final = binarize(mixed,mode="sauvola")
    cv2.imwrite(os.path.join(save_data_dir,"neg_"+pos_item),final)
    #cv2.imshow("hi",final)
    #cv2.imshow("noise",noise)
    #k = cv2.waitKey(0)
    #if(k==ord('q')):
    #    break



    
