# Traverse the required datasets and generate npy files for subsequent dataloader usage
dataset_info_train = []
dataset_info_train_input = []
dataset_info_test = []
# Read multi-level folders and get all folder paths
import os
import numpy as np
import pandas as pd
data_dir = "../ndata/xiasina/hpa"
# Read files
train_gene_dir = "train_genelist3.txt"
test_gene_dir = "test_genelist3.txt"
label_dir = "../ndata/gene_label/label"
gene_reliable = ["enhanced"]
inensity = ["inensity_level_strong","inensity_level_moderate", "inensity_level_weak"]
# inensity = ["inensity_level_strong","inensity_level_moderate"]
# inensity = ["inensity_level_strong"]
count = 0
class_dict = {
    "Nuclear membrane": 0,
    "Cytoplasm": 1,
    "Vesicles": 2,
    "Mitochondria": 3,
    "Golgi Apparatus": 4,
    "Nucleoli": 0,
    "Plasma Membrane": 1,
    "Nucleoplasm": 0,
    "Endoplasmic Reticulum": 5
}
train_genelist = []
test_genelist = []
count_class_weight = [0, 0, 0, 0, 0, 0]
# Create an empty dictionary
with open(train_gene_dir, "r") as f:
    lines = f.readlines()
    for line in lines:
        train_genelist.append(line.replace("\n", ""))
with open(test_gene_dir, "r") as f:
    lines = f.readlines()
    for line in lines:
        test_genelist.append(line.replace("\n", ""))

for i in gene_reliable:
    gene_to_category = {}
    # For each reliability folder
    # Open the label file under this reliability
    with open(os.path.join(label_dir, i + "_label" + ".txt"), "r") as f:
        # Read each line of the file
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            print(line)
            genename,category = line.split("\t")
            gene_to_category[genename] = category
    gene_reliable_dir = os.path.join(data_dir, i)
    for j in inensity:
        # For each intensity folder
        inensity_dir = os.path.join(gene_reliable_dir, j)
        for root, dirs, files in os.walk(inensity_dir):
            # Get gene na