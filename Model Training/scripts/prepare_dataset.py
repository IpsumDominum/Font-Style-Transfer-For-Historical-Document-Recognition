import pickle
import cv2
import os
from tqdm import tqdm

def get_segmentation_lines(labels,load_dir,save_dir):
    seg_labels = {}
    for index in tqdm(range(len(labels))):
        #Read the image        
        #If the image is not already segmented, read it and segment it
        if(not os.path.isfile(os.path.join(save_dir,str(len(labels[index]["bounding"])-1)+"_"+labels[index]["save_name"]))):
            DONE = False
            img = cv2.imread(os.path.join(load_dir,labels[index]["save_name"]))            
        else:
            #Otherwise pass, just record the label
            DONE = True
        for idx,b in enumerate(labels[index]["bounding"]):
            #Save the segmented image if image not already segmented
            if(DONE == False):
                seg = img[b[2]:b[3],b[0]:b[1]]
                cv2.imwrite(os.path.join(save_dir,str(idx)+"_"+labels[index]["save_name"]),seg)
                pass
            else:
                pass
            seg_labels[str(idx)+"_"+labels[index]["save_name"]] = labels[index]["text"][idx]
    return seg_labels

if __name__=="__main__": 
    pass
    
    