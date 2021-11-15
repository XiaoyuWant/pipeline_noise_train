# -*- coding: utf-8 -*-

from functools import total_ordering
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
import torch.optim as optim
import argparse, sys
import numpy as np
import datetime
import time
import warnings
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
import pandas as pd
from tqdm import tqdm
import os

from utils.Model import LoadTransforms,LoadModel,ImageFolderMy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_transforms = LoadTransforms()




def GenerateCleanDataset(model_path,out_file,dataset_path):
    model = LoadModel(name="resnet50",num_class=args.num_class,use_weight=True,weight_path=args.model_path)
    
    model=model.cuda()
    
    dataset=ImageFolderMy(root=dataset_path,transform=image_transforms['val'])
    dataloader= torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=32,
                                               num_workers=4,
                                               drop_last=False,
                                               shuffle=False)
    clean_list=[]
    name=["filepath","label"]
    with torch.no_grad():
        for imgs,labels,paths in tqdm(dataloader):
            imgs=imgs.cuda()
            labels=labels.cuda()
            outs=model(imgs)
            outputs1 = F.softmax(outs, dim=1)
            prob, pred1 = torch.max(outputs1.data, 1)
            a=(pred1==labels)
            b=torch.where(a==True)
            for ind in b[0]:
                index=paths[ind.item()]
                label=labels[ind.item()].item()
                #print(index,label)
                clean_list.append([index,label])

    df=pd.DataFrame(columns=name,data=clean_list)
    df.to_csv(out_file,index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,help="load model path of cdr training")
    parser.add_argument('--annos_name', type=str,help="annotations csv file name")
    parser.add_argument('--dataset_folder', type=str,help="all dataset path")
    parser.add_argument('--num_class', type=str,help="num_class to initiate mdoel ")

    args = parser.parse_args()

    model_path=args.model_path
    annos_name=args.annos_name
    dataset_path=args.dataset_folder

    GenerateCleanDataset(model_path,annos_name,dataset_path)

