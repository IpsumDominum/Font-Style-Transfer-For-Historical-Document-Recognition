from bs4 import BeautifulSoup
import requests 
import random
import os
from tqdm import tqdm
import asyncio
from utils import get_html
from playwright.async_api import async_playwright
import re

save_dir = "manuscript_online"

head = "http://www.aucklandcity.govt.nz"
a = "/dbtw-wpd/msonline/images/manuscripts/GLNZ/A/A1.1/web_GLNZ_A1.1.001.jpg"
a = "/dbtw-wpd/msonline/images/manuscripts/GLNZ/A/A1.1/web_GLNZ_A1.1.001.jpg"
if(os.path.isdir(save_dir)):
    pass
else:
    os.makedirs(save_dir)


srcs = []
import time
with open("manuscripts_online_srcs.txt","r") as file:
    downloads = file.readlines()
    for d in tqdm(downloads):
        sep = d.split("SEPARATOR")
        if(len(sep)==2):
            link = sep[0]
            transcription = sep[1]            
            if(link!="IMGNOTFOUND" and "pdf" not in link):
                idx = 1
                while True:                                        
                    replaced = link.replace("001","00"+str(idx))
                    img_name = replaced.split("/")[-1]
                    if(os.path.isfile(os.path.join(save_dir,img_name))):
                        idx +=1
                        break
                    try:
                        response = get_html(head+replaced)
                        if(response.headers.get('content-type')=="image/jpeg"):
                            open(os.path.join(save_dir,img_name), 'wb').write(response.content)
                        else:        
                            break
                    except requests.exceptions.ConnectionError:
                        print(head+replaced)
                    idx +=1
        else:
            continue
    exit()
