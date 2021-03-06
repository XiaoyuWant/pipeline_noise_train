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

from utils.Model import LoadTransforms,LoadModel,ImageFolderMy
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--num_class', type=int, default=0, help="No.")
parser.add_argument('--n', type=int, default=0, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='output/results_cdr/')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.2)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate')
parser.add_argument('--dataset', type=str, help='mnist, fmnist, cifar10, cifar100', default='food')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--model_type', type=str, help='[ce, ours]', default='cdr')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--weight_decay', type=float, help='l2', default=1e-3)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
parser.add_argument('--train_len', type=int, help='the number of training data', default=54000)
parser.add_argument('--folder',type=str,help="path of train data")
parser.add_argument('--output_dir',type=str,default="output",help="path of output model")
args = parser.parse_args()

print(args)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
learning_rate = args.lr
NUM_CLASS = args.num_class



# load dataset
def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    #######
    # dataloader 
    if args.dataset=='food':
        image_transforms = LoadTransforms()
        folder=args.folder
        full_dataset=ImageFolderMy(root=folder,transform=image_transforms['train'])
        train_dataset=full_dataset
        args.train_len=len(train_dataset)
        
    return train_dataset




def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
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

def getAcc(outputs,labels,batchsize):
    # ??????outputs???labels??????top1 / top3
    ret, predictions = torch.max(outputs, 1)
    correct_counts = torch.eq(predictions, labels).sum().float().item()
    acc1 = correct_counts/batchsize
    maxk = max((1,3))
    ret,predictions = outputs.topk(maxk,1,True,True)
    predictions = predictions.t()
    correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
    acc_topk = correct_counts/batchsize
    return acc1,acc_topk


def train_one_step(net, data, label, optimizer, criterion, nonzero_ratio, clip):
    time1=time.time()####
    net.train()
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()
    time2=time.time()######
    to_concat_g = []
    to_concat_v = []
    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v.append(param.data.view(-1))
    all_g = torch.cat(to_concat_g)
    all_v = torch.cat(to_concat_v)
    metric = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int(nonzero_ratio * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]

    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
            mask = mask * clip
            param.grad.data = mask * param.grad.data
    time3=time.time()#######
    
    optimizer.step()
    time4=time.time()
    optimizer.zero_grad()
    time5=time.time()
    #acc = accuracy(pred, label, topk=(1,))
    prec1,prec3=getAcc(pred,label,label.size(0))
    time6=time.time()
    #print("time:1:{}\t2:{}\t3:{}\t4:{}\t5:{}".format(time2-time1,time3-time2,time4-time3,time5-time4,time6-time5))
    return prec1,prec3, loss


def train(train_loader, epoch, model1, optimizer1, args):
    model1.train()
    train_total=0
    train_correct=0
    clip_narry = np.linspace(1-args.noise_rate, 1, num=args.num_gradual)
    clip_narry = clip_narry[::-1]
    if epoch < args.num_gradual:
        clip = clip_narry[epoch]
   
    clip = (1 - args.noise_rate)

    st_time=time.time()

    for i, (data, labels) in enumerate(train_loader):
        #ind=indexes.cpu().numpy().transpose()
        data = data.cuda()
        labels = labels.cuda()

        train_total+=1
        prec1,prec3, loss = train_one_step(model1, data, labels, optimizer1, nn.CrossEntropyLoss(), clip, clip)
        train_correct+=prec1
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Train Acc1: %.4F, Acc3: %.4F, Loss1: %.4f' 
                  %(epoch+1, args.n_epoch, i+1, args.train_len//args.batch_size, prec1, prec3,loss.item()))
            print("Time:\t{} s".format(time.time()-st_time))
            st_time=time.time()
        
      
    train_acc1=float(train_correct)/float(train_total)
    return train_acc1


# Evaluate the Model
def evaluate(test_loader, model1):
    
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    top3cnt=0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda()
            logits1 = model1(data)
            labels=labels.cuda()
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1 == labels.long()).sum()

            maxk = max((1,3))
            ret,predictions = outputs1.topk(maxk,1,True,True)
            predictions = predictions.t()
            correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
            top3cnt += correct_counts
            

        acc1 = 100 * float(correct1) / float(total1)
        acc3=top3cnt*100/float(total1)

    return acc1,acc3


def main(args):

    # Data Loader (Input Pipeline)
    print('loading dataset...')
    output_dir=args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = load_data(args)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True,
                                               pin_memory=True)
    
    
    # Define models
    print('building model...')

    ###########
    #  ?????????????????? ??????
    clf1=LoadModel(name="resnet50",num_class=args.num_class)
    optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate,momentum=0.9)
    scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)

    clf1.cuda()
    
    for epoch in range(0, args.n_epoch):
        scheduler1.step()

        clf1.train()
        
        train_acc1 = train(train_loader, epoch, clf1, optimizer1, args)

        # save model:
        torch.save(clf1.state_dict(),output_dir+"/cdr"+str(epoch)+".pth")

        # save record
        with open(output_dir+"/output.txt",'a') as f:
            text="epoch:{}\ttrain_acc:{}\n".format(epoch,train_acc1)
            f.write(text)



if __name__ == '__main__':
    best_acc = main(args)