from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import random
#from data_preprocess import *
import cv2
import matplotlib.pyplot as plt
p1 = random.randint(0,1)
p2 = random.randint(0,1)  
img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p1),
        transforms.RandomVerticalFlip(p2),
        transforms.ToTensor()])
mask_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p1),
        transforms.RandomVerticalFlip(p2),
        transforms.ToTensor()])


def random_crop(img1, img2, crop_H, crop_W):
    #print(img1.shape,img2.size)
    assert  img1.size[:2] ==  img2.size[:2]
    w, h = img2.size[:2]

    # 裁剪宽度不可超过原图可裁剪宽度
    if crop_W > w:
        crop_W = w
    # 裁剪高度

    if crop_H > h:
        crop_H = h

    # 随机生成左上角的位置
    x0 = random.randrange(0, w - crop_W + 1, 50)
    y0 = random.randrange(0, h - crop_H + 1, 50)
    #crop_1 = img1[x0:x0+crop_W,y0:y0+crop_H,:]
    crop_1 = img1.crop((x0, y0, x0+crop_W, y0+crop_H))
    crop_2 = img2.crop((x0, y0, x0+crop_W, y0+crop_H))

    return crop_1,crop_2

class RStrainDataset(Dataset):
    def __init__(self, root=None,img_transforms = img_transforms, mask_transforms = mask_transforms):
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.img_mask_dir = []
        #key_word1 = '(2)'
        key_word1 = 'label'
        for dirname in os.listdir(root):
          if not key_word1 in dirname:
            continue
          #print(dirname)
          dirname = os.path.join(root, dirname)
          #print(dirname)
          img_dir = dirname.replace("label",""),dirname
          self.img_mask_dir.append(img_dir)
        '''
        for dir_entry in os.listdir(picroot):
          if os.path.isfile(os.path.join(picroot, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            self.img_mask_dir.append((os.path.join(picroot, file_name+".tif"),os.path.join(maskroot, file_name+".png")))
        '''
        if (len(self.img_mask_dir)) == 0:
          print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_path,mask_path  = self.img_mask_dir[index]
        #print(img_path,mask_path)  
        
        img = Image.open(img_path).convert('RGB')
        
          #print(img.size)
        mask = Image.open(mask_path).convert('L')
        
        
        '''
        img = cv2.imread(img_path)
        #print(img.shape)
        img = Image.fromarray(img,mode="RGB")
        
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        #print(mask.shape)
        mask = Image.fromarray(mask,mode="L")
        '''
        img,mask = random_crop(img,mask,400,400)
        
        # transform
        if self.img_transforms is not None:
          img = self.img_transforms(img)
          #print(img.size)
        if self.mask_transforms is not None:
          mask = self.mask_transforms(mask)
        plt.subplot(1,2,1)
        plt.imshow(img.T)
        plt.subplot(1,2,2)
        plt.imshow(mask.T)
        plt.show()
        return img, mask

    def __len__(self):
        #print(len(self.img_mask_dir))
        return len(self.img_mask_dir)
val_transforms = transforms.Compose([
        transforms.ToTensor()])
  
  
  

if __name__ ==  "__main__":
  train_pic_root = r"C:/Users/banana/Documents/learn/Postgraduate_research/dl/vaihingen/train"
  #train_mask_root = "/content/drive/My Drive/pic/label_1"
  train_data_set = RStrainDataset(root = train_pic_root)
  print(train_data_set.__len__())
  train_data_set.__getitem__(1)
