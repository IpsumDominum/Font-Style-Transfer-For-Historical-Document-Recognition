import os
import pickle
from tqdm import tqdm
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image  

filename = os.path.join("data","test_labels.pkl")
labels = pickle.load(open(filename, 'rb'))
filename = os.path.join("data","vocab.pkl")
vocab = pickle.load(open(filename, 'rb'))
filename = os.path.join("data","reverse_vocab.pkl")
reverse_vocab = pickle.load(open(filename, 'rb'))
len_vocab = len(vocab)
    
def textsize(self, text, font=None, *args, **kwargs):
    """Get the size of a given string, in pixels."""
    if self._multiline_check(text):
        return self.multiline_textsize(text, font, *args, **kwargs)
    if font is None:
        font = self.getfont()
    return font.getsize(text)

import pytesseract

def get_segmentation_lines_style_transfer(labels,load_dir,save_dir):
    #Seg the sample, store in fonts/data name, then generate line which is regular font instead... 
    num_samples = 0
    formatted_bibles = {}
    regular_font = ImageFont.truetype(os.path.join("fonts","OpenSans-Regular.ttf"), 25)                
    for index in tqdm(range(len(labels))):
        #Read the image        
        #If the image is not already segmented, read it and segment it
        img = cv2.imread(os.path.join(load_dir,labels[index]["save_name"]))    
        blank = labels[index]["blank"]
        begin = labels[index]["begin"]
        text = labels[index]["text"]

        for idx,b in enumerate(labels[index]["bounding"]):
            #Save the segmented image if image not already segmented
            seg = img[b[2]:b[3],b[0]:b[1]]

            #Generate normal font (Open Sans Regular)
            bg = np.ones((50,1300))*255
            image = Image.fromarray(np.uint8(bg))
            draw = ImageDraw.Draw(image)       
            l = text[idx]
            gen_size = textsize(draw,l,font=regular_font)
            draw.text((8,10), l, font=regular_font)
            gen_img = np.array(image)[0:20+gen_size[1],0:20+gen_size[0]]
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            """
            cv2.imshow("hi",gen_img)
            k = cv2.waitKey(0)
            if(k==ord('q')):
                cv2.destroyAllWindows()
                exit()        
            break
            """
            #Save item
            cv2.imwrite(os.path.join(os.path.join(save_dir,"fonts"),str(idx)+"_"+labels[index]["save_name"]),seg)
            #Save regular font...
            cv2.imwrite(os.path.join(os.path.join(save_dir,"ground_truth"),str(idx)+"_"+labels[index]["save_name"]),gen_img)
            num_samples +=1
    print("Done...Did {} samples".format(num_samples))
if __name__=="__main__":
    get_segmentation_lines_style_transfer(labels,"samples_test",os.path.join("Style_Transfer_Data","Test"))