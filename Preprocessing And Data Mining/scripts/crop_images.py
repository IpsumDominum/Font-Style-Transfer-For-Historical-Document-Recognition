import os
import cv2
import numpy as np
import random
#Randomly crop the images and store those which contains a reasonable pixel density?(not too high not too low)

def crop_all_in_directory(MODE="MANUSCRIPT"):
    crop_size = 256
    sub_crop_size = 32
    lower_bound = 218
    upper_bound = 1050
    lap_thresh = 19900

    CRAWLED_CROPCROP_DENSITY_PARAMS = (66,2560)
    CRAWLED_CROPCROP_VAR_PARAMS = 25000
    if(MODE=="CRAWLED"):
        sub_lower_bound = 100
        sub_upper_bound = 420
    elif(MODE=="MANUSCRIPT"):
        sub_upper_bound = 8400

    save_dir = "cropped_manuscript_online"
    data_dir = "manuscript_online_binarized"

    if(os.path.isdir(save_dir)):
        pass
    else:
        os.makedirs(save_dir)
    """
    print("cleaning")
    for img_name in os.listdir(save_dir):
        os.remove(os.path.join(save_dir,img_name))
    print("removed")
    """

    kernel = np.ones((3,3),np.uint8)*0.2
    for img_name in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir,img_name))
        for i in range(img.shape[0]//crop_size):
            for j in range(img.shape[1]//crop_size):
                cropped_region = img[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size]

                """
                dilated =  cv2.morphologyEx(255-cropped_region, cv2.MORPH_GRADIENT, kernel)
                dilated = cv2.erode(255-cropped_region,kernel,iterations = 3)
                lap = cv2.Laplacian(255-cropped_region,cv2.CV_sub_crop_size4F).var()
                #Find text dense region from heuristic: dilate the inverse, the more difference in dilated the more the area.
                #Large areas = text.
                var = round(np.var(dilated.flatten()),2)
                density = round(np.sum( (255-cropped_region).flatten())/(sub_crop_size4*sub_crop_size4),2)
                #if(density>lower_bound and density <upper_bound and lap>lap_thresh):
                """
                density_score = 0
                if(MODE=="MANUSCRIPT"):
                    cropped_region = cv2.resize(cropped_region,(128,128),interpolation=3)
                else:
                    cropped_region[cropped_region>190] = 255
                for idx in range(128//sub_crop_size):
                    for jdx in range(128//sub_crop_size):
                        cropped_cropped_region = cropped_region[idx*sub_crop_size:(idx+1)*sub_crop_size,jdx*sub_crop_size:(jdx+1)*sub_crop_size].copy()
                        cropped_cropped_region[cropped_cropped_region>120] = 255
                        cropped_cropped_region_original = cropped_cropped_region.copy()
                        cropped_cropped_region = cv2.dilate(cropped_cropped_region,kernel,iterations=1)
                        cropped_cropped_region = cv2.dilate(255-cropped_cropped_region,kernel,iterations=1)
                        density = np.sum((255-cropped_cropped_region).flatten())/(sub_crop_size*sub_crop_size)
                        var = 0
                        if(MODE=="CRAWLED"):
                            if(density<66 or density >320):
                                var = 1+random.random()
                                if(density>250):
                                    density_score +=density/50
                                else:
                                    density_score += 2
                            else:
                                var = cv2.Laplacian(255-cropped_cropped_region,cv2.CV_64F).var()
                            if(var<25000):
                                density_score +=1
                        else:
                            if(density<250 or density >680):
                                if(density>730):
                                    density_score += (density-730)*20
                                else:
                                    density_score += 100
                        replacement = img_name.replace(".","_cropped"+"_"+str(i)+str(j)+".")
                        if(density!=0):
                            replacement = str(var)+".jpg"
                            replacement = str(density)+".jpg"
                            #cv2.imwrite(os.path.join(save_dir,replacement), cv2.resize(cropped_cropped_region_original,(128,128),interpolation=3))
                
                if(density_score<=sub_upper_bound):
                    replacement = img_name.replace(".","_cropped"+"_"+str(i)+str(j)+".")
                    #replacement = str(density_score)+"_"+str( round(random.random(),5))+".jpg"
                    cv2.imwrite(os.path.join(save_dir,replacement),cropped_region)
