import sys
sys.path.append("scripts")
from binarize import binarize
from crop_images import crop_all_in_directory
import os
from tqdm import tqdm

def binarize_all_in_directory(root_dir,directory,mode="otsu"):
    if(not os.path.isdir("Binarized")):
        os.makedirs("Binarized")
    if mode not in ["sauvola","otsu","nick","niblck"]:
        print("Invalid mode, choose from : " + str(["sauvola","otsu","nick","niblck"]))
    for item in tqdm(os.listdir(os.path.join(root_dir,directory))):
        binarized = binarize(cv2.imread(item),mode=mode)

if __name__ == "__main__":
    binarize_all_in_directory("Data","manuscript_online",mode="otsu")
    binarize_all_in_directory("Data","crawled",mode="otsu")
