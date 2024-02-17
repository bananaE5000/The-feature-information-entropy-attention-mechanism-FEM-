""" Parts of the U-Net model """
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt

class csAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, in_channels):
        super().__init__()
        ratio = 1
        self.w_qs = nn.Linear(1, in_channels)
        self.w_ks = nn.Linear(1, int(in_channels/ratio))
        self.w_vs = nn.Linear(in_channels, int(in_channels/ratio))
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x1):
        t = torch.var(x1,dim = [2,3])
        #t = torch.ones(x1.shape[0],x1.shape[1]).to('cuda')
        #print(t.shape)
        b,c = t.shape
        q = k = t.view(b,c,1)
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(t)
        v = torch.unsqueeze(v,dim = 2)
        #print(q.shape,k.shape,v.shape)

        attn = torch.bmm(q,k)#.transpose(1, 2))
        attn = self.softmax(attn/np.sqrt(c))
        #print(attn.shape)
        output = torch.bmm(attn , v)
        #print(output)
        output = self.sigmoid(output)
        output = torch.unsqueeze(output,dim = 3)
        #plt.plot(output.squeeze().detach().numpy())
        #plt.show()
        return output
class oneAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, in_channels):
        super().__init__()
        ratio = 1
        self.w_qs = nn.Linear(1, in_channels)
        self.w_ks = nn.Linear(1, int(in_channels/ratio))
        self.w_vs = nn.Linear(in_channels, int(in_channels/ratio))
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x1):
        #t = torch.var(x1,dim = [2,3])
        t = torch.ones(x1.shape[0],x1.shape[1]).to('cuda')
        #print(t.shape)
        b,c = t.shape
        q = k = t.view(b,c,1)
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(t)
        v = torch.unsqueeze(v,dim = 2)
        #print(q.shape,k.shape,v.shape)

        attn = torch.bmm(q,k)#.transpose(1, 2))
        attn = self.softmax(attn/np.sqrt(c))
        #print(attn.shape)
        output = torch.bmm(attn , v)
        output = self.sigmoid(output)
        output = torch.unsqueeze(output,dim = 3)
        return output
class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #t = torch.var(x,dim = [2,3])
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        #y = torch.ones(x1.shape[0],x1.shape[1]).to('cuda')
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)
class oneSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(oneSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #t = torch.var(x,dim = [2,3])
        b, c, _, _ = x.size()
        #y = self.avg_pool(x).view(b, c)
        y = torch.ones(x.shape[0],x.shape[1]).to('cuda')
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)
class CBAMAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAMAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, atten_type = '', atten = False, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels,out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.atten = atten
        self.atten_type = atten_type
        if atten == True:
          if self.atten_type == 'catten':
            #print('my')
            self.atten2 = csAttention(mid_channels)
          if self.atten_type == 'senet':
            #print('se')
            self.atten2 = SELayer(mid_channels)
          if self.atten_type =='onesenet':
            self.atten2 = oneSELayer(mid_channels)
          if self.atten_type =='cbam':
            self.atten2 = CBAMAttention(mid_channels)
          if self.atten_type == 'oneatten':
            self.atten2 =  oneAttention(mid_channels)
          if self.atten_type == 'dual':
            #print('dual')
            self.atten2 = CAM_Module(mid_channels)
    def forward(self, x):
        
        if self.atten == True:
          
          x1 = self.conv1(x)
          x2 = self.conv2(x1)
          if self.atten_type == 'dual':
            #print("dual")
            x2 = self.atten2(x2)
          else:
            #print('my')
            x2 = self.atten2(x2)*x2
        else:
          x1 = self.conv1(x)
          x2 = self.conv2(x1)
            
        return x2
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, atten, atten_type = ''):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, atten = atten, atten_type = atten_type
        )

    def forward(self, x):
        x1 = self.maxpool(x)
        #
        return self.conv(x1)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #print(1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)#, mid_channels = in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
