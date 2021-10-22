import os
from PIL import ImageFont, ImageDraw, Image  
import numpy as np
import os
import cv2
import random
import pickle
from sys import argv


def textsize(self, text, font=None, *args, **kwargs):
    """Get the size of a given string, in pixels."""
    if self._multiline_check(text):
        return self.multiline_textsize(text, font, *args, **kwargs)

    if font is None:
        font = self.getfont()
    return font.getsize(text)
            

blanks = os.listdir(os.path.join("Blanks"))

bible_original = open("kingjamesbible","r").read().split("\n")


def reformat(bible,textlength):
    bible_reformatted = []
    for line in bible:
        split_idx = 0
        while(True):
            next_line = line[split_idx:]
            if(len(next_line)>textlength):
                bible_reformatted.append(line[split_idx:split_idx+textlength])
                split_idx = split_idx +textlength
            else:
                bible_reformatted.append(next_line)
                break
    return bible_reformatted

alphabets = "abcdefghijklmnopqrstuvwxyz ,;"
alphabets = alphabets + alphabets.upper()
def split(word):
    return [char for char in word]
alphabets = split(alphabets)

def get_text_length(font):
    numpy_image = np.zeros((512,800))
    image = Image.fromarray(np.uint8(numpy_image))
    draw = ImageDraw.Draw(image)
    text_num = 0
    while(True):
        test_text = "".join(list(np.random.choice(alphabets,text_num)))
        size = textsize(draw,test_text,font=font)
        text_num +=1
        if(size[0]>400):
            break
    return text_num


labels_new = []
from tqdm import tqdm

filename = "labels.pkl"
labels = pickle.load(open(filename, 'rb'))

for font_name in tqdm(os.listdir("fonts")):
    if(font_name.endswith(".ttf") and "Anothershabby" not in font_name):
        font = ImageFont.truetype(os.path.join("fonts",font_name), 25)                
        textlength = get_text_length(font)     
        bible = reformat(bible_original,textlength)
        lines = []
        lines_bound = []       
        for label in labels:
            if(font_name in label["save_name"]):
                item = label["save_name"]
                begin = int(item[:len(item)-len(font_name)-4])
                #Get a range of text within the bible
                end = begin + 22
                #===========================
                #Create Blank Canvas
                #===========================
                blank_choice = label["blank"]#random.choice(blanks)
                bg = 255 - cv2.imread(os.path.join("Blanks",blank_choice))
                numpy_image = cv2.resize(bg,(512,800))
                image = Image.fromarray(np.uint8(numpy_image))
                draw = ImageDraw.Draw(image)                    
                #===========================
                #Iterate through the lines
                #===========================
                for idx in range(begin,end):                
                    #Create new image switching 18 lines.
                    #numpy_image = np.zeros((512,512))
                    print(idx)
                    l = bible[idx]
                    if(l!=""):
                        lines.append(l)
                        size = textsize(draw,l,font=font)
                        lines_bound.append([48,52+size[0],82+(idx%22)*25,83+(idx%22)*25+size[1]])
                    #draw.text((50, 80+ (idx%22)*25), l, font=font)
                labels_new.append({
                    "blank":blank_choice,
                    "save_name":str(begin)+font_name+".png",
                    "begin":begin,
                    "text":lines,
                    "font":font_name,
                    "bounding":lines_bound
                })
                lines = []
                lines_bound = []
                
import pickle
filename = "labels.pkl"
pickle.dump(labels_new, open(filename, 'wb')) 
exit()

for font_name in tqdm(os.listdir("fonts")):
    if(font_name.endswith(".ttf") and "Anothershabby" not in font_name):        
        all_items = os.listdir("samples_new")
        font = ImageFont.truetype(os.path.join("fonts",font_name), 25)                
        textlength = get_text_length(font)     
        bible = reformat(bible_original,textlength)
        lines = []
        lines_bound = []       
        for item in all_items:
            if(font_name in item):
                begin = int(item[:len(item)-len(font_name)-4])
                #Get a range of text within the bible
                end = begin + 22
                #===========================
                #Create Blank Canvas
                #===========================
                blank_choice = random.choice(blanks)
                bg = 255 - cv2.imread(os.path.join("Blanks",blank_choice))
                numpy_image = cv2.resize(bg,(512,800))
                image = Image.fromarray(np.uint8(numpy_image))
                draw = ImageDraw.Draw(image)                    
                #===========================
                #Iterate through the lines
                #===========================
                for idx in range(begin,end):                
                    #Create new image switching 18 lines.
                    #numpy_image = np.zeros((512,512))
                    l = bible[idx]
                    if(l!=""):
                        lines.append(l)
                        size = textsize(draw,l,font=font)
                        lines_bound.append([48,52+size[0],82+(idx%22)*25,83+(idx%22)*25+size[1]])
                    #draw.text((50, 80+ (idx%22)*25), l, font=font)
                labels.append({
                    "blank":blank_choice,
                    "save_name":str(begin)+font_name+".png",
                    "begin":begin,
                    "text":lines,
                    "font":font_name,
                    "bounding":lines_bound
                })
                lines = []
                lines_bound = []
            break
    
import pickle
filename = "labels.pkl"
pickle.dump(labels, open(filename, 'wb')) 