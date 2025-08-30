import torch
import numpy as np
from torch.utils.data import dataset
from PIL import Image,ImageFile
import torchvision.transforms as T
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
        data = self.data_list[index]
        image_path = data['path']
        img = Image.open(image_path)
        img = self.train_transforms(img)
        if self.transform:
            img = self.transform(img)
        label = data['category']
        label = torch.Tensor([label]).long()
        return img, label
        # return img, image_path
    def __len__(self):
        return len(self.data_list)

