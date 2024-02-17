import argparse
import time
import os
import json
from dataloader_p import *
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
#from palette import colorize_mask
from PIL import Image
from collections import OrderedDict
from tensorboardX import SummaryWriter
from class_names import eight_classes
 
def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    
    # dataset
    parser.add_argument('--dataset-name', type=str, default='five')
    parser.add_argument('--train-data-root', type=str, default=r'C:\Users\banana\Documents\learn\Postgraduate_research\dl\OpenEarthMap_Mini\a_train')
    parser.add_argument('--train-batch-size', type=int, default=8, metavar='N', help='batch size for training (default:8)')
    parser.add_argument('--train-crop-size', type=int, default=400, metavar='N', help='the H and W for training (default:400)')
    parser.add_argument('--train-num-classes', type=int, default=9, metavar='N', help='the H and W for training (default:9)')
    
    # output_save_path
    parser.add_argument('--experiment-start-time', type=str, default=time.strftime('%m-%d-%H_%M_%S', time.localtime(time.time())))
    
    # model
    parser.add_argument('--model', type=str, default='mod2', help='model name')
    parser.add_argument('--attentype', type=str, default='dual', help='backbone name')
    parser.add_argument('--output-stride', type=int, default=16, help='')

    # loss
    parser.add_argument('--loss-names', type=str, default='cross_entropy')
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.00000016, metavar='M', help='weight-decay (default:1e-4)')
    
    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='Adam')
    
    # learning_rate
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='M', help='')
    
    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=1, help='numbers of GPUs')
    parser.add_argument('--num_workers', type=int, default=1)

    # accuracy thoreshold 
    parser.add_argument('--best-kappa', type=float, default=0.5)
 
    args = parser.parse_args()
    #save path
    directory = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\result_OED\\"+args.attentype+'_'+args.model+"\\%s\\" % (args.experiment_start_time)
    args.directory = directory

 
    if args.use_cuda:
        print('Numbers of GPUs:', args.num_GPUs)
    else:
        print("Using CPU")
    return args
 
 
class Trainer(object):
    def __init__(self, args):
        self.args = args        
            
            
        #加载数据
        self.train_dataset = RStrainDataset(root=args.train_data_root)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.train_batch_size,
                                       num_workers=0,
                                       shuffle=True,drop_last = True,pin_memory=True)
        #print('class names {}.'.format(self.train_dataset.class_names))
        print('Number samples {}.'.format(len(self.train_loader)))
        self.num_classes = args.train_num_classes
        print("类别数：", self.num_classes)
        
        #设置损失函数
        weights = torch.ones(9) # 假设有9个类别
        weights[0] = 0 # 将第0个类别的权重设为0
        #weights[1] = 0 # 将第1个类别的权重设为0
        #weights[6] = 0 # 将第6个类别的权重设为0
        self.criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean', ignore_index=-1).cuda()#不优化第一个类别
 
        #选择使用的模型
        if args.model == 'mod1':
            from unet import UNet
            self.model = UNet(n_channels=3, n_classes=self.num_classes, atten_type = args.attentype)
        if args.model == 'mod2':
            from unet_mod2 import UNet
            self.model = UNet(n_channels=3, n_classes=self.num_classes, atten_type = args.attentype)
        if args.model == 'mod3':
            from unet_mod3 import UNet
            self.model = UNet(n_channels=3, n_classes=self.num_classes)
        if args.use_cuda:
            self.model = self.model.cuda()
            #self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        #选择优化器
        if args.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(),lr=args.base_lr,weight_decay=args.weight_decay)
        if args.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.base_lr)
        if args.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),lr=args.base_lr,momentum=args.momentum,weight_decay=args.weight_decay)
        
        #选择学习率算法
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch:0.99992, last_epoch = -1)

        
    def training(self, epoch):
        '''
        print(1)
        path_checkpoint = "/content/drive/MyDrive/segementation/checkpoint/catten_mod1/03-12-03:58:38/epoch_4800_acc_0.96771_kappa_0.92055.pth"#断点路径
        checkpoint = torch.load(path_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])#加载模型可学习参数
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(2)
        '''
        print(args.model,args.attentype)
        train_loss = average_meter.AverageMeter()
        print("len(self.train_loader)",len(self.train_loader),self.train_dataset.__len__())
        curr_iter = epoch * len(self.train_loader)
        train_conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        print(1,"*")
        for j in range(epoch):
          
          self.model.train()# 把module设成训练模式，对Dropout和BatchNorm有影响
          
          for index, data in enumerate(tbar):
              #1）
              imgs = Variable(data[0])
              masks = Variable(data[1]*255)
              if self.args.use_cuda:
                  imgs = imgs.cuda()
                  masks = masks.cuda()
                  
              #2）
              self.optimizer.zero_grad()
              
              #3）
              outputs = self.model(imgs)
              preds = torch.argmax(outputs, dim=1)
              preds = preds.data.cpu().numpy().astype(np.uint8)
              b = masks.reshape(self.args.train_batch_size,self.train_crop_size,self.train_crop_size).long()
              loss = self.criterion(outputs, b)
              
              #无关紧要的
              train_loss.update(loss, self.args.train_batch_size)
              writer.add_scalar('train_loss', train_loss.avg, curr_iter)
              #4）
              loss.backward()
              self.optimizer.step()
              self.scheduler.step()
              
              tbar.set_description('epoch {}, training loss {}, with learning rate {}.'.format(j, train_loss.avg, args.base_lr))
              #计算训练精度
              masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
              train_conf_mat += metric.confusion_matrix(pred=preds.flatten(),label=masks.flatten(),num_classes=self.num_classes)
            
          if ((j) % 100 == 0):
              
              #打印目前训练精度
              print("after ",j," eachos,the loss is ",loss)
              train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = metric.evaluate(train_conf_mat)
              writer.add_scalar(tag='train_loss_per_epoch', scalar_value=train_loss.avg, global_step=epoch, walltime=None)
              writer.add_scalar(tag='train_acc', scalar_value=train_acc, global_step=epoch, walltime=None)
              writer.add_scalar(tag='train_kappa', scalar_value=train_kappa, global_step=epoch, walltime=None)
              table = PrettyTable(["序号", "名称", "acc", "IoU"])
              a = eight_classes()
              for i in range(self.num_classes):
                  table.add_row([i, a[i], train_acc_per_class[i], train_IoU[i]])
              print(table)
              print("train_acc:", train_acc)
              print("train_mean_IoU:", train_mean_IoU)
              print("kappa:", train_kappa)
              
              #保存模型
              if train_mean_IoU > self.args.best_kappa:
                  if ((j) % 200 == 0):
                      lr = self.optimizer.param_groups[0]['lr']
                      print("lr:",lr)
                      model_name = 'epoch_%d_acc_%.5f_kappa_%.5f' % (j, train_acc, train_mean_IoU)
                      
                      checkpoint = {'epoch': epoch,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),}
                      torch.save(checkpoint, os.path.join(self.args.directory, model_name+'.pth'))
                      model_name = 'epoch_%d_acc_%.5f_kappa_%.5f' % (j, train_acc, train_mean_IoU)
                      
              #更新保存要求        
              self.args.best_kappa = train_mean_IoU
              train_conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     

if __name__ == "__main__":
    setup_seed(20)
    #the type of attention and the usage of attention
    attype = ["dual"]
    atmode = ["mod2"]
    #attype = ['catten','oneatten','senet','cbam','onesenet']
    #atmode = ["mod1"]
    
    for i in range(len(attype)):
    	for j in range(len(atmode)):
    	    torch.backends.cudnn.benchmark = True
            
    	    args = parse_args()
    	    args.model = atmode[j]
    	    args.attentype = attype[i]
            
    	    directory = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\result_OED\\"+args.attentype+'_'+args.model+"\\%s\\" % (args.experiment_start_time)
    	    args.directory = directory
            
    	    if not os.path.exists(directory):
    	        os.makedirs(directory)
    	    config_file = os.path.join(directory, 'config.json')
    	    with open(config_file, 'w') as file:
    	        json.dump(vars(args), file, indent=4)
            
    	    writer = SummaryWriter(args.directory)
            
    	    if args.use_cuda:
    	    	print(torch.cuda.get_device_name(0))
            
    	    trainer = Trainer(args)
    	    trainer.training(2802)
