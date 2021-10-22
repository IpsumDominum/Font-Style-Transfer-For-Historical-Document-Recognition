import os
import cv2
import numpy as np
import sys
sys.path.append("scripts")
from binarize import binarize
from sklearn import svm


#Load data
neg_dir = os.path.join("temp","neg")
pos_dir = os.path.join("temp","pos")

samples = []
test_samples = []

import random
from tqdm import tqdm

neg_train_count = 0
neg_test_count = 0
for idx,item in tqdm(enumerate(os.listdir(neg_dir))):
    if(random.random()<0.01):
        img = cv2.imread(os.path.join(neg_dir,item),0)
        if(random.random()>0.0):
            samples.append((img.flatten(),-1))
            neg_train_count +=1
        else:
            test_samples.append((img.flatten(),-1))
            neg_test_count +=1
                
print("Neg_Train_count: ",neg_train_count)
print("Neg_Test_count: ",neg_test_count)

pos_train_count = 0
pos_test_count = 0
for idx,item in tqdm(enumerate(os.listdir(pos_dir))):
    if(random.random()<0.01):
        img = cv2.imread(os.path.join(neg_dir,item),0)
        img = cv2.resize()
        if(random.random()>0.0):
            samples.append((img.flatten(),1))
            pos_train_count +=1
        else:
            test_samples.append((img.flatten(),1))
            pos_test_count +=1
            
print("Pos_Train_count: ",pos_train_count)
print("Pos_Test_count: ",pos_test_count)

#Shuffle array
random.shuffle(samples)
random.shuffle(test_samples)
flattened_samples = np.zeros((len(samples),128*128))
flattened_samples_test = np.zeros((len(test_samples),128*128))
sample_labels = np.zeros(len(samples))
sample_labels_test = np.zeros(len(test_samples))
#Split two arrays

for idx,item in enumerate(samples):
    flattened_samples[idx] = samples[idx][0]
    sample_labels[idx] = samples[idx][1]
for idx,item in enumerate(test_samples):
    flattened_samples_test[idx] = test_samples[idx][0]
    sample_labels_test[idx] = test_samples[idx][1]

print("done")
# fit the model
model = svm.SVC(gamma=1,verbose=True)
model.fit(flattened_samples, sample_labels)
#print(clf_weights.score(flattened_samples_test, sample_labels_test))

import pickle
filename = os.path.join("models","patch_svm.pkl")
pickle.dump(model, open(filename, 'wb'))
