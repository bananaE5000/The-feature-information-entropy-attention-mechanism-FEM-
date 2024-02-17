# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:00:33 2023

@author: banana
"""

# 导入必要的模块
import os
import PIL.Image
import matplotlib.pyplot as plt
# 定义源文件夹和目标文件夹的路径
source_folder = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\result_last\scatter_v\scatter"# 这里填写你的源文件夹的路径
target_folder = r"C:/Users\banana\Documents\learn\Postgraduate_research\dl\论文\pic_new\scatter_v" # 这里填写你的目标文件夹的路径

# 遍历源文件夹下的所有文件
for file in os.listdir(source_folder):
    # 判断文件是否是图片
    if file.endswith((".jpg", ".png", ".bmp")):
        # 打开图片
        image = PIL.Image.open(os.path.join(source_folder, file))
        # 获取图片的宽度和高度
        
        # 裁切图片
        cropped_image = image.crop((228, 164, 1740, 1297))
        #plt.imshow(cropped_image)
        #plt.show()
        # 保存图片到目标文件夹，保持原来的文件名
        cropped_image.save(os.path.join(target_folder, file),dpi = (300,300))
        # 关闭图片
        image.close()
        cropped_image.close()
