import cv2
import math
import numpy as np
import sys
import os
import time

def otsu(gray):
    """
    https://stackoverflow.com/questions/48213278/implementing-otsu-binarization-from-scratch-python
    Author: Jose A
    """
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])
        value = Wb * Wf * (mub - muf) ** 2
        #print("Wb", Wb, "Wf", Wf)
        #print("t", t, "value", value)

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    
    final_img[gray > final_thresh] = 255
    #final_img[gray < final_thresh] = 0
    return final_img

def binarize(img,mode="sauvola"):
    start_time = time.time()
    """
    Make sure that directories exist for reading and writing data
    """
    try:
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print("img not valid")
        exit()

    if(mode=="otsu"):
        out =  otsu(img)
    else:
        """
        Prepare variables
        """
        window_size = img.shape[0]//20
        #Output images 
        out = np.zeros(img.shape)        
        print("Processing")        
        #Loop over the image in a moving window manner
        for i in range(math.ceil(img.shape[0]/window_size)):
            for j in range(math.ceil(img.shape[1]//window_size)):
                #Horizontally , take the part of the image that is  (i*window_size -> (i+1)*window_size)
                #Vertically , take the part of the image that is (j*window_size -> (j+1)*window_size)
                # For instance, at i=3,j=3, for window_size 100, we have 
                # horizontally 300 : 400, vertically 300:400. So we are cropping 
                # A square that is 100x100 size.
                #  ---
                #  ---
                #  ---
                # When i =4, j=3, we are moving on to the next square indexed by 400 : 500 horizontally
                # and 300:400 vertically.
                cropped = img[i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size]

                if(mode=="niblack"):
                    # Niblack
                    # T = m + k*std
                    k = 0.05
                    T = np.mean(cropped) + k*np.std(cropped)
                    #Take the threshold
                    ret, thresh_niblack = cv2.threshold(cropped, T, 255, cv2.THRESH_BINARY)
                    #All the threshold results assigned to their correspondant window locations 
                    #in the output arrays.
                    out[i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size] = thresh_niblack

                elif(mode=="sauvola"):
                    # SAUVOLA
                    # T = m * (1-k*(1-std/R))
                    k = 0.01
                    R = 125
                    T = np.mean(cropped)*(1- k* (1 - np.std(cropped)/R))
                    #Take the threshold
                    ret, thresh_sauvola = cv2.threshold(cropped, T, 255, cv2.THRESH_BINARY)
                    #All the threshold results assigned to their correspondant window locations 
                    #in the output arrays.
                    out[i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size] = thresh_sauvola
                elif(mode=="nick"):
                    # Nick
                    # VAR = variance of the entire image compared to the window mean.
                    #       calculated by the square of each pixel - the mean of the window.
                    # T = m + k* sqrt( VAR / NP)
                    k = 0.2
                    NP = img.shape[0] * img.shape[1]
                    m = np.mean(cropped)
                    VAR = np.abs(np.sum((np.power(img.flatten(),2)-np.power(m,2)))/NP)
                    T = m + k*np.sqrt( VAR/NP)
                    #Take the threshold
                    ret, thresh_nick = cv2.threshold(cropped, T, 255, cv2.THRESH_BINARY)
                    #All the threshold results assigned to their correspondant window locations 
                    #in the output arrays.
                    out[i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size] = thresh_nick                    
    #Write out the images 
    #cv2.imwrite(os.path.join(result_dir,img_name),out)
    return out
    #print("Done.")
    #print("Took "+ str(time.time()-start_time)+" Seconds")
    
