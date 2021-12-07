
import timm
import glob
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
import pandas as pd
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def LoadModel(name="resnet50",num_class=100,use_weight=False,weight_path=""):
    if(name=="resnet50"):
        model = models.resnet50(pretrained=True)
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            # nn.LogSoftmax(dim=1)
        )
    if(name=="resnest101e"):
        model=timm.create_model('resnest101e', pretrained=True,num_classes=num_class)
    if(name=="efficientnet"):
        model=timm.create_model('efficientnet_b3', pretrained=True,num_classes=num_class)
    
    if(use_weight==True):
        model.load_state_dict(torch.load(weight_path))

    return model

def LoadTransforms():
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
    return image_transforms

def getAcc(outputs,labels,batchsize):
    # 通过outputs和labels计算top1 / top3
    with torch.no_grad():
        ret, predictions = torch.max(outputs, 1)
        correct_counts = torch.eq(predictions, labels).sum().float().item()
        acc1 = correct_counts/batchsize
        maxk = max((1,3))
        ret,predictions = outputs.topk(maxk,1,True,True)
        predictions = predictions.t()
        correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
        acc_topk = correct_counts/batchsize
        return acc1,acc_topk


def WarmUp(model,optimizer,target_lr,iter):
    # 关于warmup的复现
    if(args.local_rank==0):
        print("Warm up for iterations of:",str(iter))
    model.train()
    begin_lr=1e-6
    n_iter=0
    while(n_iter < iter):
        for i, (inputs, labels) in enumerate(train_data):
            # Set learning rate by iter
            # and update lr to warm up learning
            lr=begin_lr+n_iter*(target_lr-begin_lr)/iter
            optimizer.param_groups[0]['lr']=lr
            n_iter += 1
            if(args.local_rank==0):
                info="iter:\t{}\tlr:\t{:.5f}".format(n_iter,lr)
                print('\r', info, end='', flush=True)
            if(n_iter>iter):
                break
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            reduce_loss(loss, global_rank, world_size)
            optimizer.step()



class DatasetFromAnnos(torch.utils.data.Dataset):
    def __init__(self,csv_file,transform):
        csv=pd.read_csv(csv_file,sep=',')
        self.transform=transform
        self.img_list=csv.filepath.tolist()
        self.label_list=csv.label.tolist()
    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img="/root/dataset/refresh1000/"+self.img_list[index]
        label=self.label_list[index]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        return img,label
    def __len__(self):
        return len(self.label_list)


class ImageFolderMy(torch.utils.data.Dataset):
    def __init__(self,root,transform,imgsLimited=1000):
        classes=glob.glob(root+"/*")
        #print(classes)
        print("number of classes is:",len(classes))
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
        #print("FILE:{}".format(self.imgs[index]))
        img=self.imgs[index]
        label=self.labels[index]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        return img,label
    def __len__(self):
        return len(self.labels)

class ImageFolderGenLabel(torch.utils.data.Dataset):
    def __init__(self,root,transform,imgsLimited=1000):
        classes=glob.glob(root+"/*")
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