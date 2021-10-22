import pickle
import os
import sys
import cv2
if(os.path.isfile("checked.pkl")):
    with open('checked.pkl', 'rb') as f:
        checked = pickle.load(f)
else:
    checked = {}
count = 0
def bound_rects(img,rects,img_name):
    WIDTH = img.shape[1]
    HEIGHT = img.shape[0]
    #img = cv2.resize(img,(WIDTH,HEIGHT))
    for i,rect in enumerate(rects):                
        start = (int(rect[0]*WIDTH),int(rect[1]*HEIGHT))
        end = (int((rect[2])*WIDTH),int((rect[3])*HEIGHT))
        #start = (start[1],start[0])
        #end = (end[1],end[0])
        cropped = img[start[1]:end[1],start[0]:end[0]]
        try:
            cv2.imwrite(os.path.join(save_dir,"crop_"+str(i)+"_"+img_name),cropped)
        except cv2.error:
            print(cropped.shape)
        #img = cv2.rectangle(img,start,end,(255,0,0),2)
    return cv2.resize(img,(512,512))

#data_dir = os.path.join("../bT/Data")
save_dir = os.path.join("taming-transformers","dataset")
for k in checked.keys():
    if(len(checked[k]["rects"])!=0):
        try:
            img = cv2.imread(os.path.join(data_dir,"crawled",k))
            path = "crawled"
            cv2.imshow("img",bound_rects(img,checked[k]["rects"],k))
        except Exception as e:
            img = cv2.imread(os.path.join(data_dir,"manuscript_online",k))
            path = "manuscript_online"
            cv2.imshow("img",bound_rects(img,checked[k]["rects"],k))
        k = cv2.waitKey(1)
        if(k==ord("q")):
            break
print(count)