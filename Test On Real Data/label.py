import os
import cv2
import numpy as np
import pickle
import sys
sys.path.append("scripts")
from label_utils import (
    parse_all_labels,prepare_instructions,save_current
)
from models.fontStyleTransfer import FontStyleTransfer
from scripts.binarize import binarize

if __name__=="__main__":
    directory = os.path.join("Label_Data")
    img = None
    mode = 0
    rectangles = []
    scale = 1
    temp_rectangles = [[(0,0),(0,0)]]
    point1 = (0,0)

    model = FontStyleTransfer("74")
    nextPred = False
    #Draw preview on canvas
    def draw_rectangle(event,x,y,flags,param):
        global mouseX,mouseY,mode,temp_rectangles,rectangles,scale,img,nextPred
        x = int(x // scale)
        y = int(y // scale)
        if event == 4:
            if(mode==0):
                temp_rectangles[0][0] = (x,y)
                temp_rectangles[0][1] = (x,y)
                mode = 1
            elif(mode==1):
                rectangles = [temp_rectangles[0]]
                temp_rectangles = [[(0,0),(0,0)]]
                mode = 0
                nextPred = True
                
        if(mode==1):
            temp_rectangles[0][1] = (x,y)

    all_files = os.listdir(directory)

    """
    Prepare Window
    """
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_rectangle)
    cv2.namedWindow('crop_out')
    cv2.namedWindow('crop_out_binarized')



    idx = 0
    instructions = prepare_instructions()
    cv2.imshow("instructions",instructions)
    """
    Begin main loop
    """
    while idx < len(all_files):
        item = all_files[idx]
        img = cv2.imread(os.path.join(directory,item))
        while True:
            img_copy = img.copy()
            for rec in temp_rectangles:
                cv2.rectangle(img_copy,rec[0],rec[1],(255,0,0),1)
            for rec in rectangles:
                cv2.rectangle(img_copy,rec[0],rec[1],(0,255,0),1)
                if(nextPred==True):
                    out_crop = img_copy[rec[0][1]:rec[1][1],rec[0][0]:rec[1][0]]
                    out_crop = cv2.cvtColor(out_crop, cv2.COLOR_BGR2GRAY)
                    #out_crop = binarize(out_crop,mode="sauvola")
                    cv2.imshow('crop_out_binarized',out_crop)
                    out = model.test_on_image(out_crop)
                    out_img = np.swapaxes(out[0].cpu().detach().numpy(),0,2)
                    out_img = np.swapaxes(out_img,0,1)
                    cv2.imshow('crop_out',out_img)
                    nextPred = False

            cv2.imshow("image",cv2.resize(img_copy,( int(img.shape[1]*scale),int(img.shape[0]*scale))))
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                exit()
            elif(k==ord('z')):
                if(mode==1):
                    mode = 0
                    temp_rectangles = [[(0,0),(0,0)]]
            elif(k==ord('+')):
                scale = min(1,scale+0.1)
            elif(k==ord('-')):                                
                scale = max(0,scale-0.1)
            elif(k==ord('a')):
                idx = max((idx-1),0)
                nextPred = False
                rectangles = []
                mode = 0
                temp_rectangles = [[(0,0),(0,0)]]
                break
            elif(k==ord('d')):
                idx = min((idx+1),len(all_files)-1)
                nextPred = False
                rectangles = []
                mode = 0
                temp_rectangles = [[(0,0),(0,0)]]
                break