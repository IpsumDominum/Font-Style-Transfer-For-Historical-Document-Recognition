import os
import cv2
import numpy as np
import sys
sys.path.append("scripts")
from binarize import binarize
kernel_size = 15

def gen_patches(save_dir,data_dir):    
    for item in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir,item),0)
        #img = binarize(img,"otsu")
        #final_thresh = 150
        #img[img > final_thresh] = 255
        #img[img < final_thresh] = 0
        img_pad = np.pad(
                img, ((kernel_size//2, kernel_size//2),(kernel_size//2, kernel_size//2)), mode="constant", constant_values=255
            )
        cv2.imshow("hi",cv2.resize(img,(512,512),interpolation=0))
        k = cv2.waitKey(0)
        if(k==ord('q')):
            cv2.destroyAllWindows()
            exit()
        #Begin crop
        for i in range(img.shape[0]//kernel_size):
            for j in range(img.shape[1]//kernel_size):
                cropped_region = img_pad[i*kernel_size:(i+1)*kernel_size,j*kernel_size:(j+1)*kernel_size]
                #cv2.imshow("hi",cv2.resize(cropped_region,(512,512),interpolation=0))
                #k = cv2.waitKey(0)
                #if(k==ord('q')):
                #    cv2.destroyAllWindows()
                #    exit()
neg_dirs = [os.path.join("Extraction_Data","Neg"),os.path.join("temp","neg")]
pos_dirs = [os.path.join("Extraction_Data","Pos"),os.path.join("temp","pos")]
save_dir = os.path.join("extracted")
#Generate Patches of data
#gen_patches(os.path.join(save_dir,"Pos"),pos_dirs[0])
gen_patches(os.path.join(save_dir,"Pos"),pos_dirs[1])
gen_patches(os.path.join(save_dir,"Neg"),neg_dirs[0])
gen_patches(os.path.join(save_dir,"Neg"),neg_dirs[1])

