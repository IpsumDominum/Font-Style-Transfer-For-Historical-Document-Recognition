import cv2
import math
import numpy as np
import sys
import os
import time
sys.path.append("./eynollah")

from eynollah.qurator.eynollah.cli import eynollah_run
IMG_SAVE = os.path.join("eynollah_results","save_images")
LAYOUT_SAVE = os.path.join("eynollah_results","save_layout")
DESKEWED_SAVE = os.path.join("eynollah_results","save_deskewed")
ALL_SAVE = os.path.join("eynollah_results","save_all")

eynollah_run(
    image=os.path.join("bin_results","sauvola.PNG"),
    out=os.path.join("eynollah_results"),
    model=os.path.join("eynollah","models_eynollah"),
    save_images=IMG_SAVE,
    save_layout=LAYOUT_SAVE,
    save_deskewed=DESKEWED_SAVE,
    save_all=ALL_SAVE,
    enable_plotting=True,
    allow_enhancement=False,
    curved_line=False,
    full_layout=False,
    input_binary=True,
    allow_scaling=False,
    headers_off=True,
    log_level="DEBUG"
)
