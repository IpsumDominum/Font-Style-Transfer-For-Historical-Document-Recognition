from bs4 import BeautifulSoup
import requests 
import random
import os
from tqdm import tqdm
import asyncio
from playwright.async_api import async_playwright
import re

save_dir = "manuscript_online"

directory = "http://www.aucklandcity.govt.nz/dbtw-wpd/exec/dbtwpub.dll?AC=NEXT_BLOCK&XC=/dbtw-wpd/exec/dbtwpub.dll&BU=http%3A%2F%2Fwww.aucklandcity.govt.nz%2Fdbtw-wpd%2Fmsonline%2Findex.htm&TN=Manuscriptsonline&SN=AUTO3865&SE=1490&RN=40&MR=10&TR=0&TX=1000&ES=0&CS=1&XP=&RF=WebReport&EF=&DF=WebRecord&RL=0&EL=0&DL=0&NP=2&ID=&MF=WPEngMsg.ini&MQ=&TI=0&DT=&ST=0&IR=1761&NR=0&NB=4&SV=0&SS=1&BG=&FG=&QS=index&OEX=ISO-8859-1&OEH=ISO-8859-1"
directory = "http://www.aucklandcity.govt.nz/dbtw-wpd/exec/dbtwpub.dll?AC=PREV_RECORD&XC=/dbtw-wpd/exec/dbtwpub.dll&BU=http%3A%2F%2Fwww.aucklandcity.govt.nz%2Fdbtw-wpd%2Fmsonline%2Findex.htm&TN=Manuscriptsonline&SN=AUTO30172&SE=1496&RN=%201%20&MR=10&TR=0&TX=1000&ES=0&CS=1&XP=&RF=WebReport&EF=&DF=WebRecord&RL=0&EL=0&DL=0&NP=2&ID=&MF=WPEngMsg.ini&MQ=&TI=0&DT=&ST=0&IR=0&NR=0&NB=0&SV=0&SS=1&BG=&FG=&QS=index&OEX=ISO-8859-1&OEH=ISO-8859-1"
head = "http://www.aucklandcity.govt.nz/dbtw-wpd/exec/dbtwpub.dll?AC=NEXT_RECORD&XC=/dbtw-wpd/exec/dbtwpub.dll&BU=http%3A%2F%2Fwww.aucklandcity.govt.nz%2Fdbtw-wpd%2Fmsonline%2Findex.htm&TN=Manuscriptsonline&SN=AUTO3865&SE=1490&RN="
tail = "&MR=10&TR=0&TX=1000&ES=0&CS=1&XP=&RF=WebReport&EF=&DF=WebRecord&RL=0&EL=0&DL=0&NP=2&ID=&MF=WPEngMsg.ini&MQ=&TI=0&DT=&ST=0&IR=1911&NR=0&NB=30&SV=0&SS=1&BG=&FG=&QS=index&OEX=ISO-8859-1&OEH=ISO-8859-1"

a = "/dbtw-wpd/msonline/images/manuscripts/GLNZ/A/A1.1/web_GLNZ_A1.1.001.jpg"
def url_to_code(url):
    idx = url.find("RN")
    return url[idx:idx+10]
def code_to_url(code):
    return head + str(int(code)-2) + tail

srcs = []
start_idx = 0
with open("manuscripts_online_srcs.txt","r") as file:
    start_idx = len(file.readlines())
print(start_idx)
async def print_content(page):
    content = await page.content()
    soup = BeautifulSoup(content,"html.parser")
    src_obj = []
    for img in soup.find_all("img"):
        if("msonline" in img.get("src")):
            src_obj.append(img.get("src"))
    if(len(src_obj)==0):
        src_obj.append("IMGNOTFOUND")
    src_obj.append(soup.find("div",style="max-height:450px;overflow:auto;").text)
    with open("manuscripts_online_srcs.txt","a") as file:
        file.write(src_obj[0]+"SEPARATOR"+src_obj[1].replace("\n","NEWLINE")+"\n")

async def main():
    async with async_playwright() as p:
        print("launching browser")
        browser_type = p.webkit
        browser = await browser_type.launch()
        print("browser launched")
        page = await browser.new_page()
        for i in tqdm(range(start_idx,7828)):
            directory = code_to_url(i)
            try: 
                await page.goto(directory)                
            except Exception as e:
                with open("manuscripts_online_srcs.txt","a") as file:
                    file.write("TIMEDOUT")
            await print_content(page)
        await browser.close()

asyncio.run(main())


    
#print(code_to_url(203))
"""
print("205",url_to_code("http://www.aucklandcity.govt.nz/dbtw-wpd/exec/dbtwpub.dll?AC=NEXT_RECORD&XC=/dbtw-wpd/exec/dbtwpub.dll&BU=http%3A%2F%2Fwww.aucklandcity.govt.nz%2Fdbtw-wpd%2Fmsonline%2Findex.htm&TN=Manuscriptsonline&SN=AUTO3865&SE=1490&RN=203&MR=10&TR=0&TX=1000&ES=0&CS=1&XP=&RF=WebReport&EF=&DF=WebRecord&RL=0&EL=0&DL=0&NP=2&ID=&MF=WPEngMsg.ini&MQ=&TI=0&DT=&ST=0&IR=1911&NR=0&NB=30&SV=0&SS=1&BG=&FG=&QS=index&OEX=ISO-8859-1&OEH=ISO-8859-1"))
print("212",url_to_code("http://www.aucklandcity.govt.nz/dbtw-wpd/exec/dbtwpub.dll?AC=NEXT_RECORD&XC=/dbtw-wpd/exec/dbtwpub.dll&BU=http%3A%2F%2Fwww.aucklandcity.govt.nz%2Fdbtw-wpd%2Fmsonline%2Findex.htm&TN=Manuscriptsonline&SN=AUTO3865&SE=1490&RN=210&MR=10&TR=0&TX=1000&ES=0&CS=1&XP=&RF=WebReport&EF=&DF=WebRecord&RL=0&EL=0&DL=0&NP=2&ID=&MF=WPEngMsg.ini&MQ=&TI=0&DT=&ST=0&IR=1921&NR=0&NB=31&SV=0&SS=1&BG=&FG=&QS=index&OEX=ISO-8859-1&OEH=ISO-8859-1"))
print("1",url_to_code("http://www.aucklandcity.govt.nz/dbtw-wpd/exec/dbtwpub.dll?AC=PREV_RECORD&XC=/dbtw-wpd/exec/dbtwpub.dll&BU=http%3A%2F%2Fwww.aucklandcity.govt.nz%2Fdbtw-wpd%2Fmsonline%2Findex.htm&TN=Manuscriptsonline&SN=AUTO30172&SE=1496&RN=%201%20&MR=10&TR=0&TX=1000&ES=0&CS=1&XP=&RF=WebReport&EF=&DF=WebRecord&RL=0&EL=0&DL=0&NP=2&ID=&MF=WPEngMsg.ini&MQ=&TI=0&DT=&ST=0&IR=0&NR=0&NB=0&SV=0&SS=1&BG=&FG=&QS=index&OEX=ISO-8859-1&OEH=ISO-8859-1"))
print("2",url_to_code("http://www.aucklandcity.govt.nz/dbtw-wpd/exec/dbtwpub.dll?AC=NEXT_RECORD&XC=/dbtw-wpd/exec/dbtwpub.dll&BU=http%3A%2F%2Fwww.aucklandcity.govt.nz%2Fdbtw-wpd%2Fmsonline%2Findex.htm&TN=Manuscriptsonline&SN=AUTO30172&SE=1496&RN=0&MR=10&TR=0&TX=1000&ES=0&CS=1&XP=&RF=WebReport&EF=&DF=WebRecord&RL=0&EL=0&DL=0&NP=2&ID=&MF=WPEngMsg.ini&MQ=&TI=0&DT=&ST=0&IR=0&NR=0&NB=0&SV=0&SS=1&BG=&FG=&QS=index&OEX=ISO-8859-1&OEH=ISO-8859-1"))
print(code_to_url(203)=="http://www.aucklandcity.govt.nz/dbtw-wpd/exec/dbtwpub.dll?AC=NEXT_RECORD&XC=/dbtw-wpd/exec/dbtwpub.dll&BU=http%3A%2F%2Fwww.aucklandcity.govt.nz%2Fdbtw-wpd%2Fmsonline%2Findex.htm&TN=Manuscriptsonline&SN=AUTO3865&SE=1490&RN=201&MR=10&TR=0&TX=1000&ES=0&CS=1&XP=&RF=WebReport&EF=&DF=WebRecord&RL=0&EL=0&DL=0&NP=2&ID=&MF=WPEngMsg.ini&MQ=&TI=0&DT=&ST=0&IR=1911&NR=0&NB=30&SV=0&SS=1&BG=&FG=&QS=index&OEX=ISO-8859-1&OEH=ISO-8859-1")
"""