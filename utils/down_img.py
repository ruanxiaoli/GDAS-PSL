import csv
import os
import io
from os import path
from PIL import Image
import requests
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
urllib3.disable_warnings()

OUT_DIR = "./ndata/xiasina/hpa/enhanced/"
# OUT_DIR = "./ndata/xiasina/hpa/"
gene_dir = "./ndata/gene_label/label/"

MAX_DOWNLOAD_TRIES = 5
def parset(t):
    if t == "urinary bladder":
        return "bladder"
    else:
        return t

XML_BASE_URL = "https://www.proteinatlas.org/search/"
proxys = {
    "http": "127.0.0.1:10809",
    "https": "127.0.0.1:10809"
}
img_url_list = []

# read gene
gene_label_dict = {}
with open(os.path.join(gene_dir,"unique_genes.txt"), "r+") as f:
    lines = f.readlines()
    gene_list = [item.replace("\n", "").split(maxsplit=1)[0] for item in lines]
    # gene_list = [item.replace("\n", "").split(maxsplit=1)[0] for item in lines]
    # label_list = [item.replace("\n", "").split(maxsplit=1)[1] for item in lines]
    # for i in range(len(gene_list)):
    #     gene_label_dict[gene_list[i]] = label_list[i]
nflag = 0

# def download_and_verify_image(image_url, save_path):
#     tries = 0
#     while tries < MAX_DOWNLOAD_TRIES:
#         try:
#             # down
#             resp = requests.get(image_url, proxies=proxys,verify=False)
#             image_data = resp.content
#
#             # try to open
#             image = Image.open(io.BytesIO(image_data))
#             image.verify()
#
#             # if success
#             with open(save_path, "wb") as f:
#                 f.write(image_data)
#
#             return True
#         except Exception as e:
#             print(f"img{save_path}failed，try：{tries + 1}")
#             tries += 1
#
#     return False

import time

def download_and_verify_image(image_url, save_path, max_retries=5, delay_between_retries=5):
    _, file_extension = os.path.splitext(image_url)
    if file_extension.lower() in ['.svs', '.tif']:
        return False

    for _ in range(max_retries):
        try:
            resp = requests.get(image_url, proxies=proxys,verify=False)
            image_data = resp.content
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            with open(save_path, "wb") as f:
                f.write(image_data)

            return True
        except Exception as e:
            print(f"img{save_path}failed，try：{_ + 1}")
            time.sleep(delay_between_retries)  # sleep

    return False


#down img
def download_img_from_xml(gene_id):
    xml_file = open(path.join(OUT_DIR,"enhanced","xml",gene_id + ".xml"), "r", encoding="utf-8")
    soup = BeautifulSoup(xml_file, "xml")
    tissue_expression_all = soup.find_all("tissueExpression", {"technology": "IHC"})
    for tissue_expression in tissue_expression_all:
        data_list = tissue_expression.find_all("data")
        for data in data_list:
            organ = data.tissue.get_text()
            if organ.lower() not in ['liver', 'breast', 'prostate', 'urinary bladder']:
            # if organ.lower() in ['liver', 'breast', 'prostate', 'urinary bladder']:
                continue
            level = data.tissueCell.find("level", {"type": "intensity"})
            # lflag = level and level.get_text() in ['strong', 'moderate',"weak"]
            lflag = level and level.get_text() in ['strong']
            # lflag = level and level.get_text() in ['moderate',"weak"]
            quantity = data.tissueCell.find("quantity")
            qflag = quantity and quantity.get_text() in ['>75%', '&gt;75%']
            if lflag and qflag:
                level_dir = level.get_text()
                if not path.exists(path.join(OUT_DIR, "enhanced", "inensity_level_" + level_dir, gene_id)):
                    os.makedirs(path.join(OUT_DIR, "enhanced", "inensity_level_" + level_dir, gene_id))
                for imageUrl in data.find_all("imageUrl"):
                    save_path =path.join(OUT_DIR,"enhanced","inensity_level_"+level_dir, gene_id+"/",
                                          imageUrl.get_text().split("/", 3)[-1].replace("/", "-"))
                    # print(save_path)
                    if path.exists(save_path):
                        continue
                    success = download_and_verify_image(imageUrl.get_text(), save_path)
                    if not success:
                        print(f"img{save_path}failed，ignore")
    xml_file.close()

#
info =1
while info:
    try:
        if not path.exists(os.path.join(OUT_DIR,"enhanced","xml")):
            os.makedirs(os.path.join(OUT_DIR,"enhanced","xml"))
        if not path.exists(os.path.join(OUT_DIR,"enhanced","pictures")):
            os.makedirs(os.path.join(OUT_DIR,"enhanced","pictures"))
        number = 0
        for gene_id in tqdm(gene_list):
            number +=1
            if path.exists(os.path.join(OUT_DIR,"enhanced","xml",gene_id+".xml")):
                download_img_from_xml(gene_id)
                continue
            url = XML_BASE_URL + gene_id + "?format=xml"
            resp = requests.get(url, proxies=proxys,verify=False)
            with open(os.path.join(OUT_DIR,"enhanced","xml",gene_id + ".xml"), "w", encoding="utf-8") as f:
                f.write(resp.text)
            download_img_from_xml(gene_id)
        if number == len(gene_label_dict):
            info = 0
    except:
        print("Connection refused by the server..")
        print(f"Let me sleep for {20} seconds")
        print("ZZzzzz...")
        time.sleep(20)
        print("Was a nice sleep, now let me continue...")
        continue

