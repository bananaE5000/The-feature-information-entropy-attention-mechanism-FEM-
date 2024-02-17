# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:44:44 2023

@author: banana
"""
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
from unet import UNet
import torch
from torch.autograd import Variable
import seaborn as sns
from class_names import six_classes
import umap
import umap.plot
import numpy as np
import re
import matplotlib.font_manager as fm
    
def getindex(weight,class_type):
    if class_type == "high":
        threshold = 0.667
        #classes = weight[threshold <= weight[:]]
        index1 = [i for i in range(len(weight)) if (threshold <= weight[i])]
    elif class_type == "low":
        threshold = 0.334
        #classes = weight[weight[:]<= threshold]
        index1 = [i for i in range(len(weight)) if (weight[i] <= threshold)]
    else:
        threshold1 = 0.334
        threshold2 = 0.667
        index1 = [i for i in range(len(weight)) if (threshold1 <= weight[i] <= threshold2)]
    
    
    #index = np.isin(weight,classes)
    #index1 = np.arange(len(weight))[index]
    return index1
def umapshow(feature,class_list,weight):
    c,h,w = feature.shape
    mapper = umap.UMAP(n_neighbors=5, n_components=2,spread = 3.0).fit(feature.reshape(c,h*w))
    print(mapper.embedding_.shape)
    F_umap_2d = mapper.embedding_


    #for i in range(len(weight)):
    for idx, class_type in enumerate(class_list): # 遍历每个类别
        # 获取颜色和点型
        color = palette[idx]
        marker_list = ['^', ',', 'o', 'v', '.', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|']
        marker = marker_list[idx%len(marker_list)]

        # 找到所有标注类别为当前类别的图像索引号
        iindex = getindex(weight[0],class_type)
        
        plt.scatter(F_umap_2d[iindex, 0], F_umap_2d[iindex, 1], color=color, label=class_type,marker=marker, s=70)
        
        
    
path = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\result_v\weight_v\catten_mod1.txt"
with open(path,'r') as f:
    a = f.read()
e =  re.split("\n",a)

f = "".join(e)
b = re.split("\s{3}|\s{2}|\s{1}",a)
b = str(f).split("][")

weight = []
for i in range(len(b)):
    
    c = b[i].replace("[","").replace("]","")
    d = re.split("\s{3}|\s{2}|\s{1}",c)
    #print(d) 
    for j in range(len(d)):
        if len(d[j])<=1:
            d.remove(d[j])
    temp_num_weight = np.zeros((len(d)))
    for j in range(len(d)):
        temp_num_weight[j] = float(d[j])
    weight.append(temp_num_weight)
    #plt.plot(temp_num_weight)
    #plt.show()



img_transform = transforms.Compose([
    transforms.ToTensor()])
img_path = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\vaihingen\top\top_mosaic_09cm_area4.tif"
image = Image.open(img_path).convert("RGB")
image = np.asarray(image)[1025:1025+800,520:520+800,:]
image = Variable(img_transform(image))

image= image.unsqueeze(0)

model = UNet(n_channels=3, n_classes=6, atten_type = 'catten')
path_checkpoint = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\result_v\epoch_6400_acc_0.97593_kappa_0.93765.pth"#断点路径
checkpoint = torch.load(path_checkpoint)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
model.load_state_dict(checkpoint['model_state_dict'])#加载模型可学习参数
model.eval()
pred,feature = model(image)
pred = torch.argmax(pred, dim=1)
pred = pred.data.cpu().numpy().astype(np.uint8)
pred = pred.reshape((800,800)).astype(np.uint8)
feature1 = feature.squeeze().detach().numpy()
feature2 = feature.squeeze().detach().numpy()/np.asarray(weight[0]).reshape(-1,1,1)


class_list = ["high","med","low"]
print(class_list)

n_class = len(class_list) # 测试集标签类别数
palette = sns.color_palette("RdBu_r", n_class) # 配色方案
palette2 = sns.color_palette("Paired", n_class)

palette[1] = palette2[2]
palette[0] = palette2[0]
palette = sns.color_palette( "tab10",n_class+1)[:-1]
#plt.figure(figsize=(14, 14))
umapshow(feature2,class_list,weight)

font = fm.FontProperties(family="Times New Roman", size=12,stretch = 'ultra-expanded')
plt.legend(bbox_to_anchor=(0., -.02, 1., -.02),ncol = 3, mode="expand", borderaxespad=0.,prop = font)
plt.xticks([])
plt.yticks([])
plt.savefig(r'C:\Users\banana\Desktop\1.png', dpi=300) # 保存图像
plt.show()
'''
umapshow(feature2,class_list,weight)
plt.show()
'''