# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:35:45 2023

@author: banana
"""
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
img_path = r"C:\Users\banana\Documents\learn\Postgraduate_research\dl\OpenEarthMap_Mini\a_train\test\mask\bielefeld_68.tif"
image = Image.open(img_path).convert("RGB")
image = np.asarray(image)[200:200+800,:800,:]
image2 = image[600:800,430:630]
plt.imshow(image2)
plt.show()