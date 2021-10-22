import os
import cv2
import numpy as np
import sys
sys.path.append("scripts")
import random
from binarize import binarize
#Randomly crop the images and store those which contains a reasonable pixel density?(not too high not too low)

#import pickle
#filename = os.path.join("models","patch_svm.pkl")
#model = pickle.load(open(filename, 'rb'))

def extract_text_line(img,img_sauvola,img_raw,img_name,WRITE=False):
    crop_size = 256
    crop_height = 256
    crop_width = 256
    sub_crop_size = 16    
    """
    Iterate over the image...
    """
    img = np.pad(
                img, ((0, 0),(0, crop_width)), mode="constant", constant_values=255
            )
    img_sauvola = np.pad(
                img_sauvola, ((0, 0),(0, crop_width)), mode="constant", constant_values=255
            )

    img_raw =np.pad(
                img_raw, ((0, 0),(0, crop_width),(0,0)), mode="constant", constant_values=255 )

    kernel = np.ones((3,3),np.uint8)*0.2        

    img_comp = img.copy()
    img_comp[img_comp>120] = 255
    img_comp = cv2.dilate(img_comp,kernel,iterations=1)
    img_comp= cv2.dilate(255-img_comp,kernel,iterations=1)
    im_density = np.sum((255-img_comp).flatten())/(img.shape[0]*img.shape[1])
    if(np.var(img_comp<500)):
        im_denstiy = im_density + -300 * (1-np.var(img_comp)/500)
        kernel = np.ones((3,3),np.uint8)*0.001
    else:
        kernel = np.ones((3,3),np.uint8)*0.2        

    
    for i in range(img.shape[0]//crop_width):
        for j in range(img.shape[1]//crop_height):
            cropped_region = img[i*crop_width:(i+1)*crop_width,j*crop_height:(j+1)*crop_height]                        
            density_score = 0
            #cropped_region[cropped_region>190] = 255
            for idx in range(crop_width//sub_crop_size):
                for jdx in range(crop_height//sub_crop_size):
                    cropped_cropped_region = cropped_region[idx*sub_crop_size:(idx+1)*sub_crop_size,jdx*sub_crop_size:(jdx+1)*sub_crop_size].copy()                    
                    cropped_cropped_region_original = cropped_cropped_region.copy()
                    cropped_cropped_region[cropped_cropped_region>120] = 255
                    if(np.sum((255-cropped_cropped_region).flatten())/(sub_crop_size*sub_crop_size) <100):
                        density = np.sum((255-cropped_cropped_region).flatten())/(sub_crop_size*sub_crop_size)
                    else:
                        cropped_cropped_region = cv2.dilate(cropped_cropped_region,kernel,iterations=1)
                        cropped_cropped_region = cv2.dilate(255-cropped_cropped_region,kernel,iterations=1)
                        density = np.sum((255-cropped_cropped_region).flatten())/(sub_crop_size*sub_crop_size)
                    """
                        cv2.imshow("crop",cv2.resize(cropped_cropped_region,(512,512)) )
                        cv2.imshow("hi",cv2.resize(cropped_cropped_region_original,(512,512 )))
                        k = cv2.waitKey(0)
                        if(k==ord('q')):
                            cv2.destroyAllWindows()
                            exit()
                    """
                    if(WRITE==True):
                        replacement = img_name.replace(".","_cropped"+"_"+str(i)+str(j)+".")
                        replacement = str(density)+".jpg"
                        if( density>20 and density <im_density):
                            density_score +=1
                            #cv2.imwrite(os.path.join(temp_dir,replacement), cv2.resize(cropped_cropped_region_original,(128,128),interpolation=3))cv2.imwrite(os.path.join(temp_dir,replacement), cv2.resize(cropped_cropped_region_original,(128,128),interpolation=3))
                            img_raw = cv2.rectangle(img_raw, (j*crop_size+jdx*sub_crop_size,i*crop_size+idx*sub_crop_size), (j*crop_size+(jdx+1)*sub_crop_size,i*crop_size+(idx+1)*sub_crop_size), (0,0.1,0), 1)
                        else:   
                            #cv2.imwrite(os.path.join(save_dir,replacement), cv2.resize(cropped_cropped_region_original,(128,128),interpolation=3))
                            pass
            if(density_score>=18):
                #img_raw = cv2.rectangle(img_raw, (j*crop_size,i*crop_size), ((j+1)*crop_size,(i+1)*crop_size), (0.1,0,0), 1)
                replacement = img_name.replace(".","_cropped"+"_"+str(i)+str(j)+".")
                #cv2.imwrite(os.path.join("temp","crop",replacement),img[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size])
                #pass
    
    cv2.imshow("hi",img_raw)
    k = cv2.waitKey(0)
    if(k==ord('q')):
        cv2.destroyAllWindows()
        exit()
if __name__=="__main__":
    data_dir = os.path.join("samples")
    for item in os.listdir(data_dir):
        #if("ASTG" in item or "ASTI" in item):
        img = cv2.imread(os.path.join(data_dir,item))
        img_binarized = binarize(img,mode="otsu")
        img_sauvola = binarize(img,mode="sauvola")
        extract_text_line(img_binarized,img_sauvola,img,item,WRITE=True)
