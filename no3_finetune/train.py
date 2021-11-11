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
ImageFile.LOAD_TRUNCATED_IMAGES = True

#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 torch_ddp.py
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--world_size', type=int, help="num of processes")
parser.add_argument('--dataset_folder',type= str,default='/root/dataset/limit/', help="num of processes")
parser.add_argument('--model',type=str,default='tresnet')
parser.add_argument('--batchsize',type=int,default=32)
parser.add_argument('--csv_file_path',type=str,default="")
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--num_class',type=int,default=32)
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
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('food/runs/exp2')
def set_seed(seed):
    #必须禁用模型初始化中的任何随机性。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.set_deterministic(True)
set_seed(999)

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size

image_transforms = {
    'train':transforms.Compose([
	    #transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.Resize(size=512),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
}



# THIS IS DATASET PATH AND PARAMS
Folder=args.folder
trainDatapath=Folder+'Train'
valDatapath=Folder+'Val'
BATCH_SIZE = args.batchsize
NUM_CLASS = args.num_class
LR = args.lr
NUM_EPOCH = args.num_epoch


class DatasetFromAnnos(torch.utils.data.Dataset):
    def __init__(self,root,transform):
        csv=pd.read_csv(args.csv_file_path,sep=',')
        self.transform=transform
        self.img_list=csv.filepath.tolist()
        self.label_list=csv.label.tolist()
    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.img_list[index]
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
        

# train_dataset=ImageFolder(root=trainDatapath,transform=image_transforms['train'])
# val_dataset=ImageFolder(root=valDatapath,transform=image_transforms['val'])
# Load Dataset 
# 2-8 分割

#full_dataset=DatasetFromAnnos(root=trainDatapath,transform=image_transforms['train'])
full_dataset=ImageFolderMy(root=args.folder,transform=image_transforms['train'])
train_size=int(len(full_dataset)*0.8)
val_size=len(full_dataset)-train_size
train_dataset,val_dataset=torch.utils.data.random_split(full_dataset,[train_size,val_size])

trainsampler = DistributedSampler(train_dataset,rank=args.local_rank)
valsampler = DistributedSampler(val_dataset,rank=args.local_rank)

train_data = DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=trainsampler,num_workers=4,pin_memory=True)
val_data = DataLoader(val_dataset,batch_size=BATCH_SIZE,sampler=valsampler,num_workers=4,pin_memory=True)
print("Train size:",len(train_dataset),"; val size:",len(val_dataset))


resnet50 = models.resnet50(pretrained=True)
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 512),
    nn.ReLU(),
    nn.Linear(512, NUM_CLASS),
    # nn.LogSoftmax(dim=1)
)

###
# new_model = torch.load("food_test.pt")
# torch.save(new_model.module.state_dict(),"model_for_test.pt")


# Distributed to device
resnet50.cuda()
resnet50 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet50)
resnet50 = DDP(resnet50, device_ids=[args.local_rank], output_device=args.local_rank)

# LOSS OPTIMIZER
loss_function = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(resnet50.parameters(),lr=LR,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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
    return 0

def getAcc(outputs,labels,batchsize):
    # 通过outputs和labels计算top1 / top3
    ret, predictions = torch.max(outputs, 1)
    correct_counts = torch.eq(predictions, labels).sum().float().item()
    acc1 = correct_counts/batchsize
    maxk = max((1,3))
    ret,predictions = outputs.topk(maxk,1,True,True)
    predictions = predictions.t()
    correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
    acc_topk = correct_counts/batchsize
    return acc1,acc_topk



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
            with open(args.output_file_path+"/output.txt",'a') as f:
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
        train_data.sampler.set_epoch(epoch)
        
        ttime=time.time()
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            #print("batchsize:{}\tsingle card size:{}".format(args.batchsize,inputs.size(0)))
            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            #torch.distributed.barrier()
            loss.backward()
            optimizer.step()
            
            # New Cal of ACC
            record_freq=20
            if(i%record_freq==record_freq-1 and args.local_rank==0):
                # ret, predictions = torch.max(outputs, 1)
                # correct_counts = torch.eq(predictions, labels).sum().float().item()
                # acc1 = correct_counts/inputs.size(0)
                # maxk = max((1,3))
                # ret,predictions = outputs.topk(maxk,1,True,True)
                # predictions = predictions.t()
                # correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
                # acc_topk = correct_counts/inputs.size(0)
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
        torch.save(model, output_file_path+"/food"+str(epoch+1)+'.pt')

        scheduler.step()
        
    return model


if __name__='__main__':


    trained_model= train_and_valid(resnet50, optimizer, NUM_EPOCH)