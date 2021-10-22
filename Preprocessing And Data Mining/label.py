import os
import cv2
import numpy as np
import pickle
import sys
sys.path.append("scripts")
from label_utils import (
    parse_all_labels,prepare_instructions,save_current
)

if __name__=="__main__":
    print("Please choose between 1.'crawled' 2.'manuscript_online")
    selection = input()
    if(selection.replace(" ","")=="1"):
        print("Labeling crawled...")
    elif(selection.replace(" ","")=="2"):
        print("Labeling manuscript_online...")
    exit()
    directory = os.path.join("Data","crawled")
    img = None
    mode = 0
    rectangles = []
    scale = 1
    temp_rectangles = [[(0,0),(0,0)]]
    point1 = (0,0)

    #Draw preview on canvas
    def draw_rectangle(event,x,y,flags,param):
        global mouseX,mouseY,mode,temp_rectangles,rectangles,scale,img
        x = int(x // scale)
        y = int(y // scale)
        if event == 4:
            if(mode==0):
                temp_rectangles[0][0] = (x,y)
                temp_rectangles[0][1] = (x,y)
                mode = 1
            elif(mode==1):
                rectangles.append(temp_rectangles[0])
                temp_rectangles = [[(0,0),(0,0)]]
                mode = 0
        if(mode==1):
            temp_rectangles[0][1] = (x,y)

    all_files = os.listdir(directory)
    """
    Load all labels
    """
    try:
        with open('crawled_labels.pkl', 'rb') as f:
            labels = pickle.load(f)
    except FileNotFoundError:
        labels = ["" for _ in range(len(all_files))]
    """
    Prepare Window
    """
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_rectangle)

    idx = 0
    instructions = prepare_instructions()
    cv2.imshow("instructions",instructions)
    """
    Begin main loop
    """
    parsed_labels,already_labeled = parse_all_labels(labels)
    print(len(all_files))
    exit()
    while idx < len(all_files):
        item = all_files[idx]
        img = cv2.imread(os.path.join(directory,item))
        rectangles = parsed_labels[item]["rectangles"]    
        while True:
            img_copy = img.copy()
            for rec in temp_rectangles:
                cv2.rectangle(img_copy,rec[0],rec[1],(255,0,0),1)
            for rec in rectangles:
                cv2.rectangle(img_copy,rec[0],rec[1],(255,0,0),1)
            cv2.imshow("image",cv2.resize(img_copy,( int(img.shape[1]*scale),int(img.shape[0]*scale))))
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                rectangles,labels,temp_rectangles,parsed_labels = save_current(idx,item,directory,rectangles,labels,temp_rectangles,parsed_labels)            
                print("Saving")
                with open('labels.pkl', 'wb') as f:
                    pickle.dump(labels, f)
                print("Saved...Closing now")
                cv2.destroyAllWindows()
                exit()
            elif(k==ord('z')):
                if(mode==1):
                    mode = 0
                    temp_rectangles = [[(0,0),(0,0)]]
                else:
                    rectangles = rectangles[:len(rectangles)-1]
            elif(k==ord('d')):            
                rectangles,labels,temp_rectangles,parsed_labels = save_current(idx,item,directory,rectangles,labels,temp_rectangles,parsed_labels)            
                idx +=1
                break
            elif(k==ord('a')):
                rectangles,labels,temp_rectangles,parsed_labels = save_current(idx,item,directory,rectangles,labels,temp_rectangles,parsed_labels)
                idx -=1
                break
            elif(k==ord('+')):
                scale = min(1,scale+0.1)
            elif(k==ord('-')):                                
                scale = max(0,scale-0.1)
            
