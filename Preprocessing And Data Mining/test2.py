import pickle
import cv2
import numpy as np
import os
import easyocr

labels = open("manuscripts_online_srcs.txt","r").readlines()
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
count =0 
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2
for l in labels:
    img_path = os.path.join("Data","manuscript_online",l.split("SEPARATOR")[0].split("/")[-1])
    transcription = l.split("SEPARATOR")[1]
    if("Transcription of this record is not available" not in transcription and ".png" not in transcription and ".jpg" not in transcription):
        img = cv2.imread(img_path)
        all_results = reader.readtext(img_path)
        replaced = transcription.replace("(Transcription provided by Auckland Libraries staff)","")\
                                    .replace("[Transcription provided by Auckland Libraries staff]","")\
                                    .replace("[Transcription provided by Auckland Libraries Heritage Collections staff]","")\
                                    .replace("[Transcription provided by Auckland Libraries staff.]","").split("NEWLINE")
        for result in all_results:
            rect = result[0]
            img = cv2.rectangle(img,(int(rect[0][0]),int(rect[0][1])), (int(rect[2][0]),int(rect[2][1])),(255,0,0),1)
            img = cv2.putText(img, result[1], (int(rect[0][0]),int(rect[0][1])), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('hi',img)
        k = cv2.waitKey(0)
        if(k==ord('q')):
            break
    break



