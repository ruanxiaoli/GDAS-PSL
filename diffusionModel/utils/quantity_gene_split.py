# For validating the gene data partitioning to verify whether adding poor quality datasets affects accuracy
"""Divided into three tests: the first test uses a dataset with only strong quality data, 
the second uses a dataset with strong and moderate quality, and the third uses strong, moderate, and weak quality datasets

Save files as follows:

train_genelist1.txt
1 indicates the dataset number
"""

dataset_info = []
# Read multi-level folders and get all folder paths
import os
import random
import numpy as np
data_dir = "../ndata/xiasina/hpa"
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
train_count = 0
test_count = 0
count_class_weight = [0, 0, 0, 0, 0, 0]

# Divide genes into training and test sets at the protein level
# Create an empty list to store non-empty gene names
not_empty_genelist = []
# For each reliability folder
# Open the label file under this reliability
for i in gene_reliable:
    gene_reliable_dir = os.path.join(data_dir, i)
    for j in inensity:
        # For each intensity folder
        inensity_dir = os.path.join(gene_reliable_dir, j)
        for root, dirs, files in os.walk(inensity_dir):
            gene_name = os.path.basename(root)
            if len(files) != 0:
                not_empty_genelist.append(gene_name)
# Remove duplicates from gene names
not_empty_genelist = list(set(not_empty_genelist))

# Divide the dataset by gene names into 80% training set and 20% test set
# Generate random seed
random.seed(1234)
# Shuffle gene names
random.shuffle(not_empty_genelist)
# Training set gene names
train_genelist = not_empty_genelist[:int(len(not_empty_genelist)*0.8)]
# Test set gene names
test_genelist = not_empty_genelist[int(len(not_empty_genelist)*0.8):]

# Save training set and test set separately
with open("train_genelist"+str(len(inensity))+".txt", "w") as f:
    for gene in train_genelist:
        f.write(gene + "\n")
with open("test_genelist"+str(len(inensity))+".txt","w") as f:
    for gene in test_genelist:
        f.write(gene + "\n")
print("train_genelist.txt and test_genelist.txt have been saved successfully!")