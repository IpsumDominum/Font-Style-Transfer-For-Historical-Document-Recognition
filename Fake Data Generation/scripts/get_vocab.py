import os
import numpy as np

with open(os.path.join("data","kingjamesbible"),"r") as file:
    bible = file.read()
bible = [c for c in bible] + ["<start>","<end>"]
unique = np.unique(np.array(bible))

vocab = {}
reverse_vocab = {}
idx = 0
for char in unique:
    if(char!='\n'):
        vocab[idx] = char
        reverse_vocab[char] = idx
        idx +=1
    
import pickle
filename = os.path.join("data","vocab.pkl")
pickle.dump(vocab,open(filename, 'wb'))

filename = os.path.join("data","reverse_vocab.pkl")
pickle.dump(reverse_vocab,open(filename, 'wb'))