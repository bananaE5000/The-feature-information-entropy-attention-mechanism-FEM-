# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:19:46 2022

@author: banana
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
import mod_unet_parts as U
from torchinfo import summary

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,  bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = U.DoubleConv(n_channels, 64)
        self.down1 = U.Down(64, 128)
        self.down2 = U.Down(128, 256)
        self.down3 = U.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = U.Down(512, 1024 // factor)
        self.up1 = U.Up(1024, 512 // factor, bilinear = bilinear)
        self.up2 = U.Up(512, 256 // factor, bilinear = bilinear)
        self.up3 = U.Up(256, 128 // factor, bilinear = bilinear)
        self.up4 = U.Up(128, 64, bilinear = bilinear)
        self.outc = U.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=5)
    #print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print(total)
    summary(model, (1,3,400,400))
    model.eval()
    image = torch.randn(1, 3, 400, 400)
    with torch.no_grad():
        output = model.forward(image)
    #print(output.size())
