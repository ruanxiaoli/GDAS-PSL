#  Description: This file is used to train a classifier to predict the time step of the diffusion process.
import pandas as pd
import torch
import random
from data.data_utils import get_data, train_validation_split
from utils.diffusion_restore import Diffusion
from data.dataset_yuanlai import Diffusion_Dataset
from data.transforms import preprocess_transforms
from utils.dataset import Dataset,Dataset_test
from transformers import ResNetModel
from torch.utils.data import DataLoader
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
class TClassifier(torch.nn.Module):
    def __init__(self):
        super(TClassifier,self).__init__()
        self.resnet = ResNetModel.from_pretrained("./path/local/directory",local_files_only=True)
        self.resnet.requires_grad=False
        self.head  = torch.nn.Sequential(torch.nn.Linear(2048,100),torch.nn.ReLU(),torch.nn.Linear(100,1))
    def forward(self,x):
        outs1 = self.resnet(x)['pooler_output'].squeeze()
        return self.head(outs1)+1

import random
from tqdm import tqdm
def randts(size = 150, max = 500):
    return torch.LongTensor([random.randint(0,max) for i in range(size)]).cuda()

def randomcompose(img,img2):
    if random.randint(0,2) != 1:
        return randomcompose(img2,img)
    mask = torch.ones(img.size()).cuda()
    for i in range(len(mask)):
        a,b,c,d = randomregion()
        mask[i,:,a:b,c:d] = 0
    mask2 = torch.ones(img.size()).cuda() - mask
    return mask * img + mask2 * img2

def randomregion():
    l = random.randint(30,180)
    w = random.randint(30,180)
    l1 = random.randint(0,256-l)
    w1 = random.randint(0,256-w)
    return l1,l1+l,w1,w1+w

if __name__=='__main__':
    train_dataset = Dataset("./utils/train_classifier.npy", transform=None)
    trainloader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              prefetch_factor=3)

    # val dataloader
    val_dataset = Dataset_test("./utils/test_classifier.npy")
    valloader = DataLoader(val_dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            prefetch_factor=3)
    # image_data = pd.read_csv('../experiments/HQ_paths_cleaned').file_name.tolist()
    # train_data, val_data = train_validation_split(image_data,['NIO_DS_100'])
    # trainloader = torch.utils.data.DataLoader(Diffusion_Dataset(data = train_data, img_root='/nfs/turbo/umms-tocho-ns/root_srh_db',image_transforms=preprocess_transforms(256)),num_workers=8,shuffle=True,batch_size=150)
    # valloader = torch.utils.data.DataLoader(Diffusion_Dataset(data = val_data, img_root='/nfs/turbo/umms-tocho-ns/root_srh_db',image_transforms=preprocess_transforms(256)),num_workers=8,shuffle=False,batch_size=150)
    model = TClassifier().cuda()
    criterion = torch.nn.MSELoss()
    diffuser = Diffusion(timesteps = 1000, beta_schedule='cosine')
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)
    best_loss = 99999999
    for epoch in range(500):
        totalloss = 0.0
        totals = 0
        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            # Randomly added t
            ts = randts(len(batch['image']))
            # Simulate different degrees of damage
            img = diffuser.q_sample(batch['image'].cuda(), ts)
            img2 = diffuser.weak_q_sample(batch['image'].cuda(), ts)
            # Randomly compose together
            img = randomcompose(img, img2)
            # Use three channels for training
            inimg = img
            # Here model refers to resnet50
            out = model(inimg)
            # Calculate loss with two t values
            loss = criterion(out.squeeze(1), ts.float())
            totalloss += loss.item()
            loss.backward()
            optimizer.step()
            # totals is the total number of batches, training data is 197
            totals += 1
        print("epoch " + str(epoch) + " train loss " + str(totalloss / totals))
        with torch.no_grad():
            totals = 0
            totalloss = 0.0
            for batch in tqdm(valloader):
                ts = randts(len(batch['image']))
                img = diffuser.q_sample(batch['image'].cuda(), ts)
                img2 = diffuser.weak_q_sample(batch['image'].cuda(), ts)
                img = randomcompose(img, img2)
                inimg = img
                out = model(inimg)
                loss = criterion(out.squeeze(1), ts.float())
                totalloss += loss.item()
                totals += 1
            print("epoch " + str(epoch) + " val loss " + str(totalloss / totals))
            if totalloss / totals < best_loss:
                best_loss = totalloss / totals
                torch.save(model, 'classifiercosine_' + str(best_loss)+str('_') + str(epoch) + '.pt')