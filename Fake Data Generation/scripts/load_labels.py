import pickle
import cv2
import os

"""
Load saved labels
"""
filename = "labels.pkl"
labels = pickle.load(open(filename, 'rb'))
save_dir = "samples"

"""
Load image with labels
Press Q to quit
"""
def view_image_with_labels(index):
    #Read the image
    img = cv2.imread(os.path.join(save_dir,labels[index]["save_name"]))

    #Put the bounding boxes
    for idx,rect in enumerate(labels[index]["bounding"]):
        img = cv2.rectangle(img,(rect[0],rect[2]),(rect[1],rect[3]),(255,0,0),1)
        print(labels[index]["text"][idx])
        #Show the image
        cv2.imshow("img",img)
        k = cv2.waitKey(0)
        if(k==ord('q')):
            return

def clean_directory():
    all_dir = os.listdir(save_dir)
    all_names = []    
    for i in range(len(labels)):
        name = labels[i]["save_name"]
        all_names.append(name)
    for i in all_dir:
        if i not in all_names:
            print(i)

if __name__=="__main__":
    view_image_with_labels(0)
    #clean_directory()
