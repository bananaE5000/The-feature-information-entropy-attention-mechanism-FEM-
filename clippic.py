# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:02:40 2023

@author: banana
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#OEMbielefeld 68
#[700:700+300,415:415+300,:]
#OEMkoeln 15
#[850:850+150,130:130+150,:]
#v20
#[900:900+400,150:150+400,:]
#v2
#[650:650+450,135:135+450,:]
'''
atten_type = ['catten1','cbam1','fre1','oneatten1','onesenet1','senet1','unet3']
#atten_type = ['catten1']
for i in range(len(atten_type)):
    
    img_path = r"C:/Users/banana/Documents/learn/Postgraduate_research/dl/result_last/result_OEM/predict_" + atten_type[i] + "/4/koeln_15_l.tif"
    image = Image.open(img_path).convert("RGB")
    image = np.asarray(image)[850:850+150,130:130+150,:]
    #plt.imshow(image)
    #plt.show()

    Image.fromarray(image).save(r'C:/Users\banana\Documents\learn\Postgraduate_research\dl\论文\pic_new\OEM_2\\'+atten_type[i]+'.tif')
'''
'''
atten_type = ['image','mask']
for i in range(len(atten_type)):
'''
    #img_path = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\OpenEarthMap_Mini\a_train\test\\" + atten_type[i] + "\\koeln_15.tif"
'''
    image = Image.open(img_path).convert("RGB")
    
    image = np.asarray(image)[850:850+150,130:130+150,:]
    #plt.imshow(image)
    #plt.show()
'''
#    Image.fromarray(image).save(r'C:\Users\banana\Documents\learn\Postgraduate_research\dl\论文\pic_new\OEM_2\\'+atten_type[i]+'.tif')

    


'''
atten_type = ['catten1','cbam1','fre1','oneatten1','onesenet1','senet1','unet3']
#atten_type = ['catten1']

for i in range(len(atten_type)):
    
    img_path = r"C:/Users/banana/Documents/learn/Postgraduate_research/dl/result_last/result_v/predict_" + atten_type[i] + "/3/top_mosaic_09cm_area2_l.tif"
    image = Image.open(img_path).convert("RGB")
    image = np.asarray(image)[650:650+450,135:135+450,:]
    #plt.imshow(image)
    #plt.show()

    Image.fromarray(image).save(r'C:/Users\banana\Documents\learn\Postgraduate_research\dl\论文\pic_new\v_2\\'+atten_type[i]+'.tif')

'''
atten_type = ['mask']
for i in range(len(atten_type)):

    img_path = r"C:/Users/banana/Documents/learn/Postgraduate_research/dl/vaihingen/labeling_Vaihingen/top_mosaic_09cm_area2_new-L.tif"

    image = Image.open(img_path).convert("RGB")
    
    image = np.asarray(image)[650:650+450,135:135+450,:]
    #plt.imshow(image)
    #plt.show()

    Image.fromarray(image).save(r'C:/Users/banana/Documents/learn/Postgraduate_research/dl/论文/pic_new\v_2/'+atten_type[i]+'.tif')
