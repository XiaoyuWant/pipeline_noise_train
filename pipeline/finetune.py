import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# System
import os
import time
import warnings
import glob
warnings.filterwarnings("ignore")
# Tools
import argparse
import numpy as np
from tqdm import tqdm

import random
from PIL import Image
from PIL import ImageFile
import pandas as pd
from utils.Model import LoadTransforms,DatasetFromAnnos,ImageFolderMy,getAcc,LoadModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 torch_ddp.py
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--world_size', type=int, help="num of processes")
parser.add_argument('--dataset_folder',type= str,default='/root/dataset/limit/', help="num of processes")
parser.add_argument('--model',type=str,default='resnet50')
parser.add_argument('--batchsize',type=int,default=32)
parser.add_argument('--csv_file_path',type=str,default="")
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--num_class',type=int,default=128)
parser.add_argument('--num_epoch',type=int,default=80)
parser.add_argument('--output_file_path',type=str,default="output")

args = parser.parse_args()
world_size=args.world_size
# dist.init_process_group(backend='nccl', init_method='env://')
dist.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:8989',rank=args.local_rank,world_size=args.world_size)
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()
print("RANK:",global_rank)

if not os.path.exists(args.output_file_path):
    os.mkdir(args.output_file_path)

torch.backends.cudnn.benchmark=True

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size





# THIS IS DATASET PATH AND PARAMS

BATCH_SIZE = args.batchsize
NUM_CLASS = args.num_class
LR = args.lr
NUM_EPOCH = args.num_epoch


        
image_transforms = LoadTransforms()
# train_dataset=ImageFolder(root=trainDatapath,transform=image_transforms['train'])
# val_dataset=ImageFolder(root=valDatapath,transform=image_transforms['val'])
# Load Dataset 
# 2-8 分割

#full_dataset=DatasetFromAnnos(root=trainDatapath,transform=image_transforms['train'])
full_dataset=DatasetFromAnnos(csv_file=args.csv_file_path,transform=image_transforms['train'])
train_size=int(len(full_dataset)*0.8)
val_size=len(full_dataset)-train_size
train_dataset,val_dataset=torch.utils.data.random_split(full_dataset,[train_size,val_size])

trainsampler = DistributedSampler(train_dataset,rank=args.local_rank)
valsampler = DistributedSampler(val_dataset,rank=args.local_rank)
train_batch_sampler = torch.utils.data.BatchSampler(
        trainsampler, BATCH_SIZE, drop_last=True)
train_data = DataLoader(train_dataset,batch_sampler=train_batch_sampler,num_workers=16,pin_memory=True)
val_data = DataLoader(val_dataset,batch_size=BATCH_SIZE,sampler=valsampler,num_workers=16,pin_memory=True)
print("Train size:",len(train_dataset),"; val size:",len(val_dataset))


resnet50 = LoadModel(name=args.model,num_class=NUM_CLASS)


# Distributed to device
resnet50.cuda()
# resnet50 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet50)
resnet50 = DDP(resnet50, device_ids=[args.local_rank], output_device=args.local_rank)

# LOSS OPTIMIZER
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(),lr=LR,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)




def ValidModel(model,epoch):
        valid_loss = 0.0
        with torch.no_grad():
            model.eval()
            T_count=0
            V_count=0
            V_k_count=0
            for j, (inputs, labels) in enumerate(val_data):
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs=model(inputs)
                loss = loss_function(outputs, labels)
                loss=loss.mean()

                valid_loss += loss.item() * inputs.size(0)
                # TOP1
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = torch.eq(predictions, labels).sum().float().item()
                V_count += correct_counts
                acc = correct_counts/inputs.size(0)

                # TOP5
                maxk = max((1,3))
                ret,predictions = outputs.topk(maxk,1,True,True)
                predictions = predictions.t()
                correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
                V_k_count += correct_counts
                acc_topk = correct_counts/inputs.size(0)
                #print("Val Loss for {} : {:.5f}\t Top-1 Acc {}%\t Top-3 Acc {}%".format(i,loss,acc*100,acc_topk*100))
                #记录loss
                if(j%100==99):

                    info="Val Loss for {} : {:.5f}\t Top-1 Acc {}%\t Top-3 Acc {}%".format(j,loss,acc*100,acc_topk*100)
                    print('\r',info,end=' ',flush=True)
                    #writer.add_scalar("LOSS",loss,global_step=i+epoch*len(train_data))
            loss_of_val=valid_loss*world_size/len(val_dataset)
            top1_of_val=V_count*world_size/len(val_dataset)
            top3_of_val=V_k_count*world_size/len(val_dataset)
            print("VAL:{}\tTop1:{:.2f}%\tTop3:{:.2f}%\tL:{:.5f}".format(
                        epoch,top1_of_val*100,top3_of_val*100,loss_of_val
                    ))
            # 保存记录
            with open(args.output_file_path+"/finetune.txt",'a') as f:
                text="Epoch{}\tTop1:\t{}\tTop3:\t{}\n".format(epoch,top1_of_val*100,top3_of_val*100 )
                f.write(text)
    
def train_and_valid(model, optimizer, epochs=25):

    # WARMUP
    #WarmUp(model,optimizer,LR,20000)
    for epoch in range(epochs):
        #epoch_start = time.time()
        if(args.local_rank==0):
            print("Epoch: {}/{}".format(epoch+1, epochs))
            print("This epoch is {} iterations".format(len(train_data)))
        model.train()
        # 更改trainloader_sampler
        # train_data.sampler.set_epoch(epoch)
        
        ttime=time.time()
        record_freq=20
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.cuda()
            labels = labels.cuda()
            #print("batchsize:{}\tsingle card size:{}".format(args.batchsize,inputs.size(0)))
            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            torch.distributed.barrier()
            loss.backward()
            optimizer.step()
            
            # New Cal of ACC
            
            if(i%record_freq==record_freq-1 and args.local_rank==0):

                acc1,acc_topk=getAcc(outputs,labels,inputs.size(0))
                #etaTime=(time.time()-ttime)*(len(train_data)-i)/record_freq # not accurate
                loss=loss.mean()
                ETAtime=(1-(i/len(train_data)))*(time.time()-ttime)*(len(train_dataset)/BATCH_SIZE)/record_freq/60
                info="{}/{}\tTop1:{:.2f}%\tTop3:{:.2f}%\tL:{:.5f}\ttime:{:.2f}S\tETA:{:.2f}Min".format(
                    epoch,i,acc1*100,acc_topk*100,loss,time.time()-ttime,ETAtime
                )
                print('\r',info,end=' ',flush=True)
                #print(info)
                ttime=time.time()

        # 测试结果
        ValidModel(model,epoch)
        torch.save(model.module.state_dict(), args.output_file_path+"/food"+str(epoch+1)+'.pt')

        scheduler.step()
        
    return model


if __name__=='__main__':


    trained_model= train_and_valid(resnet50, optimizer, NUM_EPOCH)