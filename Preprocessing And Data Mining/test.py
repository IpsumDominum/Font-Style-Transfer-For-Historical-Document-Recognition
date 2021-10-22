import pickle
import cv2
import numpy as np
import os

labels = pickle.load(open("labels.pkl","rb"))

for item in labels:
    path = os.path.join("samples",item["save_name"])
    img = cv2.imread(path)
    for rect in item["bounding"]:
        img = cv2.rectangle(img,(rect[0],rect[2]),(rect[1],rect[3]),(255,0,0),1)
    cv2.imshow("img",img)
    k = cv2.waitKey(0)
    if(k==ord('q')):
        cv2.destroyAllWindows()
        break