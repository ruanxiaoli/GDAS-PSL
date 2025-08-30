# Used to build custom dataset for pretraining
import torch
import numpy as np
from torch.utils.data import dataset
from PIL import Image
import torchvision.transforms as T
import random

import os

def get_gene_name_from_path(path):
    # Use os module's split and splitext functions to split the path
    path_parts = os.path.splitext(path)[0].split('/')
    # Gene name is the second-to-last part
    gene_name = path_parts[-2]
    return gene_name

class Dataset(dataset.Dataset):
    def __init__(self, data_info_path,transform=None):
        super().__init__()
        self.transform = transform
        self.data_list = np.load(data_info_path, allow_pickle=True).tolist()
        trans_list = [
            T.Resize([512, 512]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
        self.train_transforms = T.Compose(trans_list)

    def __getitem__(self, index):
        while True:
            try:
                data = self.data_list[index]
                image_path = data['path']
                img = Image.open(image_path)
                img = self.train_transforms(img)
                if self.transform:
                    img = self.transform(img)
                label = data['category']
                label = torch.Tensor([label]).long()
                return img, label
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Trying another one.")
                index = random.randint(0, len(self.data_list) - 1)

    def __len__(self):
        return len(self.data_list)


class Dataset_test(dataset.Dataset):
    def __init__(self, data_info_path,transform=None):
        super().__init__()
        self.transform = transform
        self.data_list = np.load(data_info_path, allow_pickle=True).tolist()
        trans_list = [
            T.Resize([512, 512]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
        self.train_transforms = T.Compose(trans_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path = data['path']
        img = Image.open(image_path)
        img = self.train_transforms(img)
        if self.transform:
            img = self.transform(img)
        label = data['category']
        label = torch.Tensor([label]).long()
        gene_name = get_gene_name_from_path(image_path)
        return img, label,gene_name

    def __len__(self):
        return len(self.data_list)