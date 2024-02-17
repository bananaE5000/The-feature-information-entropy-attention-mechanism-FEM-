import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from unet import UNet
#from unet_mod2 import UNet
from torch.autograd import Variable
import torch
import os
import pandas as pd
from PIL import Image
import cv2 as cv
from collections import OrderedDict
import torch.nn as nn
import tifffile as tiff
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from config import Config
import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
c = Config()
img_transform = transforms.Compose([
    transforms.ToTensor()])
mod = 2
def parse_args():
    parser = argparse.ArgumentParser(description="膨胀预测")
    parser.add_argument('--test-data-root', type=str, default=r'/home/test/duan_segmentation/data_OEM/a_train/test/image')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N', help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--attentype', type=str, default="dual", help='backbone name')
    parser.add_argument("--model-path", type=str, default=r"/home/test/duan_segmentation/checkpoint_OEM/dual_mod2/10-17-07:25:20/epoch_2800_acc_0.95028_kappa_0.82160.pth")
    parser.add_argument("--pred-path", type=str, default="")
    args = parser.parse_args()
    return args
k = "2"
 
class Inference_Dataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.img_mask_dir = []
        self.transforms = transforms
        key_word1 = '(1)'
        for dirname in os.listdir(root_dir):
          if not key_word1 in dirname:
            continue
          dirname = os.path.join(root_dir, dirname)
          self.img_mask_dir.append(dirname)
        self.img_mask_dir = []
        
    def __len__(self):
        return len(self.img_mask_dir)
 
    def __getitem__(self, idx):
        filename = self.csv_file.iloc[idx, 0]
        # print(filename)
        image_path = os.path.join(self.root_dir, filename)
        # image = np.asarray(Image.open(image_path))  # mode:RGBA
        # image = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)  # PIL(RGBA)-->cv2(BGRA)
        image = Image.open(image_path).convert('RGB')
 
        # if self.transforms:
        #     print('transforms')
        image = self.transforms(image)
 
        return image
 
def out_mask_to_color_pic(mask, palette_file='/home/test/duan_segmentation/deeplearncode/Palette.json'):
    assert len(mask.shape) == 2
    with open(palette_file, 'r') as fp:
        text = json.load(fp)
    color_pic = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    for i in tqdm(range(0, mask.shape[0])):
        for j in range(0, mask.shape[1]):
            assert str(mask[i,j]) in list(text.keys())
            color_pic[i,j,:] = text[str(mask[i,j])]
    return color_pic
 
def reference():
    args = parse_args()
    torch.cuda.empty_cache()
    #dataset = Inference_Dataset(root_dir=args.test_data_root, transforms=img_transform)
    #dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)
    #model = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=16, pretrained=False, _print=True)
    if mod == 1:
      from unet import UNet
      model = UNet(n_channels=3, n_classes=9, atten_type = args.attentype)
    if mod == 2:
      from unet_mod2 import UNet
      model = UNet(n_channels=3, n_classes=9, atten_type = args.attentype)
    if mod == 3:
      from unet_model import UNet
      model = UNet(n_channels=3, n_classes=9)
    #state_dict = torch.load(args.model_path, map_location='cpu')
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    '''
    
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model_path).items()})
    model = model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model_list = []
    model_list.append(model)
    model.eval()
    img_path = []
    for pic in os.listdir(args.test_data_root):
        #if 'aachen' in pic:
            print(pic)
            img_path.append(args.test_data_root + '/' + pic)
    stride = c.size_train[0]//2 
    for n in range(len(img_path)):
        print(n)
        path = img_path[n]
        print(path.split('/')[-1][:-8])
        #load the image
        image = Image.open(path).convert("RGB")
        image = np.asarray(image)
        h,w,_ = image.shape
        padding_h = (h//stride + 1 ) * stride
        padding_w = (w//stride + 1 ) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        rgb = np.zeros((h,w,3),dtype=np.uint8)
        for i in range (0,h):
            for j in range (0,w):
                x=h-i-1
                y=w-j-1
                rgb[i,j,:]=image[i,y,:] #镜像翻转
                
            
 
        print("padding_h-h",padding_h-h)
        print("padding_w",padding_w-w)
        padding_img[0:h,w:padding_w,:] = rgb[:,0:(padding_w-w),:]
        rgb1 = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        for p in range (0,padding_h):
            for q in range (0,padding_w):
                x=padding_h-p-1
                y=padding_w-q-1
                rgb1[p,q,:]=padding_img[x,q,:] #镜像翻转
        padding_img[h:padding_h,0:padding_w,:] = rgb1[(padding_h-h):2*(padding_h-h),:,:]
        #padding_img = padding_img.astype("float") / 255.0
 
        print ('src:',padding_img.shape)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+c.size_train[0], j*stride: j*stride+c.size_train[1], :]
                ch,cw,_ = crop.shape
                
                if ch != c.size_train[0] or cw != c.size_train[1]:
                    print ('invalid size!')
                    print(crop.shape)
                    continue
                    
                #crop = np.expand_dims(crop, axis=0)#增加维度
                #print 'crop:',crop.shape
                # pred = model.predict_classes(crop,verbose=2)
                crop = img_transform(crop)
                crop = Variable(crop)
                #print(crop)
                crop = crop.cuda()
                crop = crop.unsqueeze(0)
                #print(crop.shape)
                outputs = model(crop)
                #image = snapshot_forward(model, crop, model_list, zeros)
                pred = torch.argmax(outputs, dim=1)
                pred = pred.data.cpu().numpy().astype(np.uint8) #index of max-channel 
                '''
                pred = model.predict(crop,verbose=2)
                pred = pred[0]
                '''
                #pred = np.reshape(pred, (1, c.size_train[1] * c.size_train[0], c.n_label))
                #pred = np.argmax(pred, axis=2)
                # print(pred.shape)
 
                #pred = labelencoder.inverse_transform(pred[0])  
                #print (np.unique(pred))  
                pred = pred.reshape((c.size_train[0],c.size_train[1])).astype(np.uint8)
                #print 'pred:',pred.shape
                mask_whole[i*stride:i*stride+c.size_train[0],j*stride:j*stride+c.size_train[1]] = pred[:,:]
 
 
        out = out_mask_to_color_pic(mask_whole[0:h,0:w])
        out1 = mask_whole[0:h,0:w]
        c.check_folder('/home/test/duan_segmentation/result_OEM/predict_' + args.attentype + str(mod) + '/'+str(k))
        #Image.fromarray(out1).save('/content/drive/MyDrive/segementation/result/predict_catten2/'+path.split('/')[-1][:-4]+'_label.tif', dpi = c.dpi)
        Image.fromarray(out).save('/home/test/duan_segmentation/result_OEM/predict_' + args.attentype + str(mod) + '/'+str(k)+'/'+path.split('/')[-1][:-4]+'_l.tif', dpi = c.dpi)
 
if __name__ == '__main__':
    reference()
