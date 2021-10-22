import pickle
import os

directory = os.path.join("static","data")
checked = {}
for idx,item in enumerate(os.listdir(directory)):
    checked[item] = {"rects":[],"index":idx,"invalid":False,"contributors":[]}

with open('checked.pkl', 'wb') as f:
    pickle.dump(checked, f)