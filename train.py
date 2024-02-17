import argparse
import time
import os
import json
from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
#from unet_mod2 import UNet
#from unet import UNet

from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict
from tensorboardX import SummaryWriter
from class_names import six_classes
 
def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    # dataset
    parser.add_argument('--dataset-name', type=str, default='five')
    parser.add_argument('--train-data-root', type=str, default=r'/home/test/duan_segmentation/data_V/vaihingen/train/')
    parser.add_argument('--val-data-root', type=str, default=r'/content/drive/MyDrive/segementation/second/top')
    parser.add_argument('--train-batch-size', type=int, default=2, metavar='N', help='batch size for training (default:16)')
    parser.add_argument('--val-batch-size', type=int, default=1, metavar='N', help='batch size for testing (default:16)')
    # output_save_path
    parser.add_argument('--experiment-start-time', type=str, default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))
    parser.add_argument('--save-pseudo-data-path', type=str, default='/content/drive/My Drive/checkpoint6_7')
    # augmentation
    parser.add_argument('--base-size', type=int, default=800, help='base image size')
    parser.add_argument('--crop-size', type=int, default=800, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')
    # model
    parser.add_argument('--model', type=str, default='mod1', help='model name')
    parser.add_argument('--attentype', type=str, default='catten', help='backbone name')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--n-blocks', type=str, default='3, 4, 23, 3', help='')
    parser.add_argument('--output-stride', type=int, default=16, help='')
    parser.add_argument('--multi-grids', type=str, default='1, 1, 1', help='')
    parser.add_argument('--deeplabv3-atrous-rates', type=str, default='6, 12, 18', help='')
    parser.add_argument('--deeplabv3-no-global-pooling', action='store_true', default=False)
    parser.add_argument('--deeplabv3-use-deformable-conv', action='store_true', default=False)
    parser.add_argument('--no-syncbn', action='store_true', default=False, help='using Synchronized Cross-GPU BatchNorm')
    # loss
    parser.add_argument('--loss-names', type=str, default='cross_entropy')
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.00000016, metavar='M', help='weight-decay (default:1e-4)')
    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='Adam')
    # learning_rate
    parser.add_argument('--base-lr', type=float, default=0.01, metavar='M', help='')
    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=1, help='numbers of GPUs')
    parser.add_argument('--num_workers', type=int, default=1)
    # validation
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--val', action='store_true', default=False)
 
    parser.add_argument('--best-kappa', type=float, default=0.7)
 
    parser.add_argument('--total-epochs', type=int, default=12, metavar='N', help='number of epochs to train (default: 120)')
    parser.add_argument('--start-epoch', type=int, default=1, metavar='N', help='start epoch (default:0)')
 
    parser.add_argument('--resume_path', type=str, default=None)
 
    args = parser.parse_args()
    directory = r"/home/test/duan_segmentation/checkpoint_v/"+args.attentype+'_'+args.model+"/%s/" % (args.experiment_start_time)
    args.directory = directory

 
    if args.use_cuda:
        print('Numbers of GPUs:', args.num_GPUs)
    else:
        print("Using CPU")
    return args
 
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
 
class Trainer(object):
    def __init__(self, args):
        self.args = args
        resize_scale_range = [float(scale) for scale in args.resize_scale_range.split(',')]
 
        self.train_dataset = RStrainDataset(root=args.train_data_root)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.train_batch_size,
                                       num_workers=0,
                                       shuffle=True,drop_last = True,pin_memory=True)
        #print('class names {}.'.format(self.train_dataset.class_names))
        print('Number samples {}.'.format(len(self.train_loader)))
        
        self.num_classes = 6
        print("类别数：", self.num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-1).cuda()
 
        n_blocks = args.n_blocks
        n_blocks = [int(b) for b in n_blocks.split(',')]
        atrous_rates = args.deeplabv3_atrous_rates
        atrous_rates = [int(s) for s in atrous_rates.split(',')]
        multi_grids = args.multi_grids
        multi_grids = [int(g) for g in multi_grids.split(',')]
 
        if args.model == 'mod1':
            #self.model = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=16, pretrained=False, _print=True)
            from unet import UNet
            self.model = UNet(n_channels=3, n_classes=6, atten_type = args.attentype)

        if args.model == 'mod2':
            #self.model = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=16, pretrained=False, _print=True)
            from unet_mod2 import UNet
            self.model = UNet(n_channels=3, n_classes=6, atten_type = args.attentype)
        if args.model == 'mod3':
            #self.model = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=16, pretrained=False, _print=True)
            from unet_model import UNet
            self.model = UNet(n_channels=3, n_classes=6)

 
 
        if args.use_cuda:
            self.model = self.model.cuda()
            #self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
 
        # SGD不work，Adadelta出奇的好？
        if args.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(),lr=args.base_lr,weight_decay=args.weight_decay)
 
        if args.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.base_lr)
 
        if args.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=model.parameters(),lr=args.base_lr,momentum=args.momentum,weight_decay=args.weight_decay)
 
        self.max_iter = args.total_epochs * self.train_dataset.__len__()
        self.save_pseudo_data_path = args.save_pseudo_data_path

 
 
    def training(self, epoch):
        '''
        print(1)
        path_checkpoint = "/content/drive/MyDrive/segementation/checkpoint/catten_mod1/03-12-03:58:38/epoch_4800_acc_0.96771_kappa_0.92055.pth"#断点路径
        checkpoint = torch.load(path_checkpoint)
        print(2)
        self.model.load_state_dict(checkpoint['model_state_dict'])#加载模型可学习参数
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(3)
        '''
        print(args.model,args.attentype)
        train_loss = average_meter.AverageMeter()
        print("len(self.train_loader)",len(self.train_loader),self.train_dataset.__len__())
        curr_iter = epoch * len(self.train_loader)
        
        milestones= [3600,7200]
        #lr = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=0.2, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch:0.9999, last_epoch = -1)
        #print(self.optimizer.state_dict()["param_groups"])
        train_conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        print(1,"*")
        for j in range(epoch):
          
          self.model.train()# 把module设成训练模式，对Dropout和BatchNorm有影响
          for index, data in enumerate(tbar):
 
            # assert data[0].size()[2:] == data[1].size()[1:]
            # data = self.mixup_transform(data, epoch)
            imgs = Variable(data[0])
            masks = Variable(data[1]*255)
 
            if self.args.use_cuda:
              imgs = imgs.cuda()
              masks = masks.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            # torch.max(tensor, dim)：指定维度上最大的数，返回tensor和下标
            preds = torch.argmax(outputs, dim=1)
            preds = preds.data.cpu().numpy().astype(np.uint8)
            b = masks.reshape(2,400,400).long()
            loss = self.criterion(outputs, b)
 
            train_loss.update(loss, self.args.train_batch_size)
            writer.add_scalar('train_loss', train_loss.avg, curr_iter)
            loss.backward()
            self.optimizer.step()
            scheduler.step()
            tbar.set_description('epoch {}, training loss {}, with learning rate {}.'.format(j, train_loss.avg, args.base_lr))
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            train_conf_mat += metric.confusion_matrix(pred=preds.flatten(),label=masks.flatten(),num_classes=self.num_classes) 
            
          if ((j) % 200 == 0):
            print("after ",j," eachos,the loss is ",loss)
            train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = metric.evaluate(train_conf_mat)
            writer.add_scalar(tag='train_loss_per_epoch', scalar_value=train_loss.avg, global_step=epoch, walltime=None)
            writer.add_scalar(tag='train_acc', scalar_value=train_acc, global_step=epoch, walltime=None)
            writer.add_scalar(tag='train_kappa', scalar_value=train_kappa, global_step=epoch, walltime=None)
            table = PrettyTable(["序号", "名称", "acc", "IoU"])
            a = six_classes()
            for i in range(self.num_classes):
              table.add_row([i, a[i], train_acc_per_class[i], train_IoU[i]])
            print(table)
            print("train_acc:", train_acc)
            print("train_mean_IoU:", train_mean_IoU)
            print("kappa:", train_kappa)
            if train_mean_IoU > self.args.best_kappa:
              if ((j) % 400 == 0):
                lr = self.optimizer.param_groups[0]['lr']
                print("lr:",lr)
                model_name = 'epoch_%d_acc_%.5f_kappa_%.5f' % (j, train_acc, train_mean_IoU)
              
                checkpoint = {'epoch': epoch,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),}
                torch.save(checkpoint, os.path.join(self.args.directory, model_name+'.pth'))
 
                model_name = 'epoch_%d_acc_%.5f_kappa_%.5f' % (j, train_acc, train_mean_IoU)
              
                #torch.save(self.model.state_dict(), os.path.join(self.args.directory, model_name+'.pth'))
            self.args.best_kappa = train_mean_IoU
            train_conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     

 
if __name__ == "__main__":

 
    '''
    tbar = tqdm(trainer.train_loader)
    for j in range(31):
      for i, data in enumerate(tbar):
         #trainer.pre_compute_W(i, data)
         trainer.pre_compute_W(i, data)
'''
    #args = parse_args()
    # 设置随机数种子
    setup_seed(20)
    attype = ["unet"]
    atmode = ["mod3"]
    #attype = ['catten','oneatten','senet','cbam','onesenet']
    #atmode = ["mod1"]
    for i in range(len(attype)):
    	for j in range(len(atmode)):
    	    torch.backends.cudnn.benchmark = True
    	    args = parse_args()
    	    args.model = atmode[j]
    	    args.attentype = attype[i]
    	    directory = r"/home/test/duan_segmentation/checkpoint_v/"+args.attentype+'_'+args.model+"/%s/" % (args.experiment_start_time)
    	    args.directory = directory
    	    if not os.path.exists(directory):
    	        os.makedirs(directory)
    	    config_file = os.path.join(directory, 'config.json')
    	    with open(config_file, 'w') as file:
    	        json.dump(vars(args), file, indent=4)
    	    print(args.model,args.attentype,args.directory)
    	    writer = SummaryWriter(args.directory)
    	    trainer = Trainer(args)
    	    if args.use_cuda:
    	    	print(torch.cuda.get_device_name(0))
    	    trainer.training(6402)
