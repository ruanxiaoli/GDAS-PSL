# import sys
# import os
import os
from torchvision.utils import save_image
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy
from typing import TextIO
import json
import yaml
import argparse
import datetime
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
from utils.dataset import Dataset
# from RSCD.utils.dataset import Dataset
from utils.dataset import Dataset_test
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data.data_utils import rescale_channels, get_third_channel
from torchvision.utils import save_image
from torchsummary import summary
from classifier import TClassifier
from pathlib import Path
from models.model import Unet
from models.vmunet.vmunet import VMUNet
from models.model_utils import save_model
from data.data_utils import get_data, train_validation_split

# from data.dataset import Diffusion_Dataset

from data.transforms import preprocess_transforms,preprocess_numpy
# from utils.diffusion import Diffusion
from utils.diffusion_restore import Diffusion
# from noiseestimate import NonParametricStepPredictor
from runner import diffusion_runner
#from ensamble import Ensamble
import numpy as np
torch.random.manual_seed(23)
#### OPTIONS ###################################################################
# get config file
def parse_args() -> TextIO:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    parser.add_argument('-i',
                        '--id',
                        type=int,
                        required=False,
                        default=-1,
                        help='parallelid')
    args = parser.parse_args()
    return args.config, args.id


cf_fd,paraid = parse_args()


def main():
    cmd_input: TextIO = cf_fd
    if cmd_input.name.endswith(".json"):
        return json.load(cf_fd)
    elif cmd_input.name.endswith(".yaml") or cf_fd.name.endswith(".yml"):
        return yaml.load(cf_fd, Loader=yaml.FullLoader)

device = 'cuda:0'
config = main()


#### INSTANTIATE DIFFUSION OBJECT ##############################################
#
diffusion = Diffusion(restore_timesteps=config['diffusion']['restore_timesteps'],
                      timesteps=config['diffusion']['timesteps'],
                      beta_schedule=config['diffusion']['beta_schedule'],
                      image_size=config['data']['image_size'])

# #### GET DATALOADERS #########################################################
# train dataloader
train_dataset = Dataset("./utils/train_recover.npy", transform=None)
train_loader = DataLoader(train_dataset,
                              # batch_size=config['train']['batch_size'],
                              batch_size=1,
                              shuffle=False,
                              num_workers=5,
                              pin_memory=True,
                              prefetch_factor=3)
# #### GET MODELS ##############################################################
if config['model']['model_type'] == 'VMUNet':
    model = VMUNet(
        num_classes=config['model']['num_classes'],
        input_channels=config['model']['input_channels'],
        depths=config['model']['depths'],
        depths_decoder=config["model"]["depths_decoder"],
        drop_path_rate=config['model']['drop_path_rate'],
    )
if config['model']['model_type'] == 'unet':
        model = Unet(dim=config['data']['image_size'],
                    channels=config['model']['in_channels'],
                    init_dim=config['model']['init_dim'],
                    out_dim=config['model']['out_dim'],
                    dim_mults=config['model']['dim_mults'],
                    with_time_emb=config['model']['with_time_emb'],
                    resnet_block_groups=config['model']['resnet_block_groups'],
                    use_convnext=config['model']['use_convnext'],
                    convnext_mult=config['model']['convnext_mult'],
    #                cond=config['model']['cond']).float()
                     )

if config['model']['model_type'] == 'imagen':
    pass

model.load_state_dict(torch.load(config['inference']['model_ckpt'],map_location='cpu'))
model.eval()

print(sum(p.numel() for p in model.parameters()) / 1e6)
model = model.to(device)

# Distribute the models to the device
#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)
#    print(f"Using {torch.cuda.device_count()} GPUs.")
# summary(model, input_size=(config['model']['in_channels'],
#                         config['data']['image_size'],
#                         config['data']['image_size']))

##### GET OPTIMIZER ############################################################
optimizer = Adam(model.parameters(), lr=config['train']['lr'])
# sunglist = []
# lines = open('SungHQList').readlines()
# for line in lines:
#     sunglist.append(int(line[:-1]))
#print(sunglist)
# #### TRAIN MODELS ############################################################
# mark the time for saving results and models
# Get current time and date
now = datetime.datetime.now()
now = f'{str(now.day)}-{str(now.month)}-{str(now.year)}'
# Load classifier model
classifier = torch.load(config['inference']['classifier'],map_location='cpu').to(device)
# Iterate through data, i is batch index (which batch), batch is the batch data
data_paths = []
restore_img = []
begins_img = []
for idx,batch in enumerate(train_loader):
    if paraid >= 0:
        i = paraid
        # If batch index is less than i times 10000 or greater than or equal to i times 10000 plus 10000, skip current loop.
        if idx < i * 10000 or idx >= i * 10000 + 10000:
            continue
    print('At '+str(idx))
    begins = []
    skipped = False
    inimg2 = batch['image'].float().to(device)
    data_path = batch['image_path'][0]
    print(data_path)
    data_paths.append(data_path)
    outs = [inimg2.cpu().numpy()]
    # Maximum number of inference times for one image, here it is 15 times
    for j in range(config['inference']['max_rounds']):
        # inimg = torch.zeros(1,3,256,256).to(device)
        # Input original image: inimg
        inimg = inimg2
        # Predict a t
        begin = classifier(inimg).long().item()
        # If t is greater than the preset maximum value, set t to the maximum value (ensure each sample can be processed by the optimal t)
        if begin > config['inference']['largethreshold']:
            begin = config['inference']['largethreshold']
        # j=0 means no sampling is needed and set skipped to true
        # First prediction
        elif j==0 and  config['inference']['largeonly']:
            skipped = True
            break
        # No sampling needed for first prediction
        if j==0 and begin < config['inference']['skip_threshold']:
            # Prevent backward diffusion
            print('skipped '+str(idx)+' because '+str(begin))
            skipped = True
            break
        #     Sample to a qualifying begin, then sample data under this predicted t value
        begins.append(begin)
        if begin < config['inference']['recalibrate_threshold'] or j == config['inference']['max_rounds']-1:
            # Sampling
            outs1 = diffusion.sample(model,inimg2,1,begin=begin+3)
            inimg2 = torch.FloatTensor(outs1[-1]).to(device)
            # Add output value
            outs.append(outs1[-1])
            break
        #     Exit loop after sampling qualifying t?
        outs1 = diffusion.sample(model,inimg2,1,begin=begin+1,step_limit = config['inference']['recalibrate_steps'])
        inimg2 = torch.FloatTensor(outs1[-1]).to(device)
        outs.append(outs1[-1])
#    if skipped:
#        continue
    outs_tensors = []
    if begins:
        begins_img.append(begins[-1])
    else:
        begins_img.append(0)
    for img in outs:
        # print(f"the shape of img is{img.shape}")  # 
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.squeeze(img_tensor)  # 
        outs_tensors.append(img_tensor)
    restore_img.append(outs_tensors[-1])
    # save images
    results_folder = Path(config['inference']['result_path'])
    results_folder.mkdir(parents=True,exist_ok=True)
    results_folder_raw = Path(config['inference']['result_path_raw'])
    results_folder_raw.mkdir(parents=True,exist_ok=True)
    save_image(outs_tensors, results_folder / f'sample-{idx}-{begins}.png', nrow=10)

## Iterate over data_paths and restore_img
#for path, img in zip(data_paths, restore_img):
#    # Create new folder path
#    print(path)
#    if "inensity_level_weak" in path:
#        new_folder_path = path.replace('inensity_level_weak', 'inensity_level_weak_restore')
#    else:
#        new_folder_path = path.replace('inensity_level_moderate', 'inensity_level_moderate_restore')
#    print(new_folder_path)
#    new_folder_path = os.path.dirname(new_folder_path)  # Get the directory path
#
#    # Create the folder if it does not exist
#    os.makedirs(new_folder_path, exist_ok=True)
#
#    # Save the image to the new folder path with a unique name
#    save_image(img, os.path.join(new_folder_path, os.path.basename(path)))
#    print(f"image_{os.path.join(new_folder_path, os.path.basename(path))}_have_saved")

#def save_image(img, path, begin):
#    # Extract the base name of the file path
##    base_name = os.path.basename(path)
#    # Remove the file extension
#    path_without_ext = os.path.splitext(path)[0]
#    # Append a valid file extension
#    new_file_name = f"{path_without_ext}_{begin}.jpg"
#    # Save the image with the new file name
#    img.save(os.path.join(new_folder_path, new_file_name))
#
#count = 0
## Iterate over data_paths and restore_img
#print(len(data_paths))
#print(len(restore_img))
#for path, img,begin in zip(data_paths, restore_img,begins_img):
#    # Create new folder path
#    if 'inensity_level_weak' in path:
#        new_folder_path = path.replace('inensity_level_weak', 'inensity_level_weak_restore')
#    elif 'inensity_level_moderate' in path:
#        new_folder_path = path.replace('inensity_level_moderate', 'inensity_level_moderate_restore')
#    elif 'inensity_level_strong' in path:
#        new_folder_path = path.replace('inensity_level_strong', 'inensity_level_strong_restore')
#    else:
#        new_folder_path = None
#        print("failed")
#    count +=1
#    new_folder_path = os.path.dirname(new_folder_path)  # Get the directory path
#
#    # Create the folder if it does not exist
#    os.makedirs(new_folder_path, exist_ok=True)
#
#    # Save the image to the new folder path with a unique name
##    save_image(img, os.path.join(new_folder_path,f"{os.path.basename(path)}_{begin}"))
#    save_image(img, os.path.join(new_folder_path, os.path.basename(path)),begin)
#    print(count)


def save_images(img, path, begin):
    # Extract the base name of the file path
    base_name = os.path.basename(path)
    # Remove the file extension
    path_without_ext = os.path.splitext(base_name)[0]
    # Append 'begin' and '.jpg' extension
    new_file_name = f"{path_without_ext}_{begin}.jpg"
    # Save the image with the new file name
    save_image(img,os.path.join(new_folder_path, new_file_name))
#    img.save(os.path.join(new_folder_path, new_file_name))

count = 0
# Iterate over data_paths and restore_img
print(len(data_paths))
print(len(restore_img))
for path, img, begin in zip(data_paths, restore_img, begins_img):
    # Create new folder path based on certain conditions in the path
    if 'inensity_level_weak' in path:
        new_folder_path = path.replace('inensity_level_weak', 'inensity_level_weak_restore')
    elif 'inensity_level_moderate' in path:
        new_folder_path = path.replace('inensity_level_moderate', 'inensity_level_moderate_restore')
    elif 'inensity_level_strong' in path:
        new_folder_path = path.replace('inensity_level_strong', 'inensity_level_strong_restore')
    else:
        new_folder_path = None
        print("Incorrect path")
    
    count += 1
    new_folder_path = os.path.dirname(new_folder_path)  # Get the directory path

    # Create the folder if it does not exist
    os.makedirs(new_folder_path, exist_ok=True)

    # Save the image to the new folder path with a unique name
    save_images(img, path, begin)
    print(count)