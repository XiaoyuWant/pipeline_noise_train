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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ImageFolderMy(torch.utils.data.Dataset):
    def __init__(self,root,transform,imgsLimited=1000):
        classes=glob.glob(root+"/*")
        #print(classes)
        #classes=classes[:1000]
        self.transform=transform
        self.imgs=[]
        self.labels=[]
        for i in range(len(classes)):
            one=classes[i]
            imgs=glob.glob(one+'/*.jpg')
            if(len(imgs)>imgsLimited):
                imgs=imgs[:imgsLimited]
            #print("img len:",len(imgs))
            labels=[i for _ in range(len(imgs))]
            #print("img len:",len(labels))
            self.imgs+=imgs
            self.labels+=labels
    def __getitem__(self,index):
        img=self.imgs[index]
        imgname=img
        label=self.labels[index]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        return img,label,imgname
    def __len__(self):
        return len(self.labels)
class DatasetFromAnnos(torch.utils.data.Dataset):
    def __init__(self,root,transform):
        csv=pd.read_csv("/root/userfolder/Exp/DDP_Classification_pytorch/CDR/annos.csv",sep=',')
        self.csv=csv
    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.csv.iloc[index]["filepath"]
        label=self.csv.iloc[index]["label"]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        return img,label
    def __len__(self):
        return len(print(len(csv)))
image_transforms = {
            'train':transforms.Compose([
                #transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ]),
            'val':transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        }
def GenerateCleanDataset(model_path,out_file,dataset_path):
    model = tv_models.resnet50(pretrained=True)
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASS),
            nn.LogSoftmax(dim=1)
        )
    model.load_state_dict(torch.load(model_path))
    model=model.cuda()
    dataset=ImageFolderMy(root=dataset_path,transform=image_transforms['val'])
    dataloader= torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=32,
                                               num_workers=4,
                                               drop_last=False,
                                               shuffle=True,
                                               pin_memory=True)
    clean_list=[]
    name=["filepath","label"]
    with torch.no_grad():
        for imgs,labels,names in tqdm(dataloader):
            imgs=imgs.cuda()
            labels=labels.cuda()
            outs=model(imgs)
            outputs1 = F.softmax(outs, dim=1)
            prob, pred1 = torch.max(outputs1.data, 1)
            a=(pred1==labels)
            b=torch.where(a==True)
            for ind in b[0]:
                index=names[ind.item()]
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

    args = parser.parse_args()

    model_path=args.model_path
    annos_name=args.annos_name
    dataset_path=args.dataset_folder

    GenerateCleanDataset(model_path,annos_name,dataset_path)

