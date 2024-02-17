# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:06:58 2023

@author: banana
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
#from unet_mod2 import UNet
#from unet import UNet

from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from PIL import Image
from unet import UNet
import matplotlib.pyplot as plt


def out_mask_to_color_pic(mask, palette_file=r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\deeplearncode1\Palette1.json"):
    assert len(mask.shape) == 2
    with open(palette_file, 'r') as fp:
        text = json.load(fp)
    color_pic = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    for i in tqdm(range(0, mask.shape[0])):
        for j in range(0, mask.shape[1]):
            assert str(mask[i,j]) in list(text.keys())
            color_pic[i,j,:] = text[str(mask[i,j])]
    return color_pic

img_transform = transforms.Compose([
    transforms.ToTensor()])
320
625
img_path = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\vaihingen\top\top_mosaic_09cm_area4.tif"
image = Image.open(img_path).convert("RGB")
image = np.asarray(image)[1025:1025+800,520:520+800,:]
image = Variable(img_transform(image))

image= image.unsqueeze(0)
#print(crop.shape)

model = UNet(n_channels=3, n_classes=6, atten_type = 'catten')
path_checkpoint = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\result_v\epoch_6400_acc_0.97593_kappa_0.93765.pth"#断点路径
checkpoint = torch.load(path_checkpoint)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
model.load_state_dict(checkpoint['model_state_dict'])#加载模型可学习参数
model.eval()
pred = torch.argmax(model(image), dim=1)
pred = pred.data.cpu().numpy().astype(np.uint8)
pred = pred.reshape((800,800)).astype(np.uint8)
out = out_mask_to_color_pic(pred)
plt.imshow(out[:400,:400,:])
plt.show()
print(1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])