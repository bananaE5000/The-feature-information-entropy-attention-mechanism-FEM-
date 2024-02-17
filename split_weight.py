# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:48:22 2023

@author: banana
"""

import numpy as np
import matplotlib.pyplot as plt 
import re
import matplotlib.font_manager as fm
import seaborn as sns
def getindex(weight,class_type):
    if class_type == "heigh":
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
attentype = ['catten']
#attentype = ['onesenet','catten','senet','fre','cbam','oneatten']
for k in range(len(attentype)):
    
 path = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\result_last\scatter_oem\\"+attentype[k]+"_mod1.txt"
 with open(path,'r') as f:
     a = f.read()
 e =  re.split("\n",a)

 f = "".join(e)
 b = re.split("\s{3}|\s{2}|\s{1}",a)
 b = str(f).split("][")

 weight = []
 for i in range(len(b)):
    
    c = b[i].replace("[","").replace("]","").replace(" ",',')
    d = re.split(",,,|,,|,",c)#.replace(" ","")
    #print(d) 
    #print(d)
    print(len(d))
    dd = []
    for j in range(len(d)):
        
        if len(d[j])>=1:
            dd.append(d[j])
    d = dd
    temp_num_weight = np.zeros((len(d)))
    for j in range(len(d)):
        temp_num_weight[j] = float(d[j])
    weight.append(temp_num_weight)
 print(np.var(weight[3]))
 # 创建一个三维坐标轴
 plt.figure(figsize=(12,8),dpi=300)
 ax = plt.axes(projection='3d')
 # 遍历每个时间点
 ax.view_init(20, 310)
 for i in range(len(weight)):
    # 获取当前时间点的曲线数据
    z = np.asarray(weight[len(weight)-1-i])
    x = np.linspace(0,len(z),len(z))
    #xb = np.stack([x, b], axis=0).T
    # 用时间点作为x坐标
    y = np.full_like(z, len(weight)-1-i)
    # 在三维坐标轴上绘制曲线
    ax.plot(x, y, z)
 # 设置坐标轴的标签
 #ax.set_xlabel('time')
 #ax.set_ylabel('y')
 #ax.set_zlabel('z')
 # 显示图例
 #ax.set_aspect('auto', adjustable='box', anchor='C')
 ax.set_xticks(range(0,512,100))
 ax.set_zlim(0,1)
 ax.set_yticks([])

 font = fm.FontProperties(family="Times New Roman", size=12,stretch = 'ultra-expanded')
 #ax.legend(bbox_to_anchor=(0., .02, 1., .02),ncol = 4, mode="expand", borderaxespad=0.,prop = font)
 #ax.set_figheight(6)
 #ax.set_figwidth(8)
 # 显示图形
 path1 = r'C:\Users\banana\Desktop\\'+attentype[k]+'_oem.png'
 print(path1)
 #plt.savefig(path1,dpi = 300)
 plt.show()
 


