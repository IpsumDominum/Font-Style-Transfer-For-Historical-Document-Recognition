import os
import cv2
import numpy as np
from collections import defaultdict
import pickle
def save_current(idx,item,directory,rectangles,labels,temp_rectangles,parsed_labels):
    parsed_labels[item]["rectangles"] = rectangles
    to_save = directory+"SEP"+item+"BEGINRECTANGLES"
    for rec in rectangles:
        to_save += str(rec[0][0])+","+str(rec[0][1])+","+str(rec[1][0])+","+str(rec[1][1])+"SEP"
    labels[idx] = to_save + "\n"
    rectangles = []
    temp_rectangles = [[(0,0),(0,0)]]
    mode = 0
    return rectangles,labels,temp_rectangles,parsed_labels

def prepare_instructions():
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    instructions = np.zeros((800,512,3))
    color = (255, 0, 0)
    instructions = cv2.putText(instructions, 'Press Q to save and quit', (50,50), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    color = (0, 255, 0)
    instructions = cv2.putText(instructions, 'Press Z to undo', (50,100), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    color = (0, 255, 0)
    instructions = cv2.putText(instructions, 'Press D to move forward', (50,150), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    color = (0, 255, 0)
    instructions = cv2.putText(instructions, 'Press A to move backward', (50,200), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    color = (255, 255, 255)
    instructions = cv2.putText(instructions, 'Press + to scale up', (50,250), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    instructions = cv2.putText(instructions, 'Press - to scale down', (50,300), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    
    return instructions

def parse_label(label):
    parsed = {}
    parsed["directory"] = label.split("BEGINRECTANGLES")[0].split("SEP")[0]
    parsed["img"] = label.split("BEGINRECTANGLES")[0].split("SEP")[1]
    parsed["rectangles"] = []
    for rec in label.split("BEGINRECTANGLES")[1].split("SEP"):
        r = rec.split(",")
        if(len(r)==4):
            r = list(map(int,r))
            parsed["rectangles"].append([(r[0],r[1]),(r[2],r[3])])
    return parsed

def parse_all_labels(labels):
    parsed_labels = defaultdict(lambda:{
        "directory":"",
        "img:":"",
        "rectangles":[],
    })
    already_labeled = [] 
    for label in labels:
        try:
            parsed = parse_label(label)
            parsed_labels[parsed["img"]] = parsed
            already_labeled.append(parsed["img"])
        except IndexError:
            pass
    return parsed_labels,already_labeled