import cv2
import math
import numpy as np
import sys
import os
import time
sys.path.append("./eynollah")
from Louloudis.Implementation import performLouloudisSegmentation
lou = performLouloudisSegmentation(os.path.join("bin_results","sauvola.PNG"))
print("==")
print(lou)
print("==")
