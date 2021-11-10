import torch.nn as nn
from torchvision import models
import os
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt
'''
resnet-18 layer3--->thre=0.005
resnet-18 layer2--->thre=0.01
resnet-18 layer1--->thre=0.02
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_gpu = True
showfeature = True
h = 1500
w = 3010

def getImage(filepath):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img_data = cv2.imread(filepath)
    img_data = cv2.resize(img_data, (w, h), interpolation=cv2.INTER_CUBIC)
    img_data = np.array(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)).astype(np.uint8)
    return transform(img_data)


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    # modify the forward function
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def getHash(f1,f2,f0, thre=None):
    f1 = np.mean(f1, axis=(1, 2))
    f2 = np.mean(f2, axis=(1, 2))
    f0 = np.mean(f0, axis=(1, 2))
    dist1 = np.abs(f0-f1)
    dist2 = np.abs(f0-f2)
    if thre is None:
        thre = (np.min(dist1)+np.max(dist1)+np.min(dist2)+np.max(dist2))/4.0
    hash1 =[]
    print(thre)
    for i in range(dist1.shape[0]):
       if dist1[i] > thre:
           hash1.append(1)
       else:
           hash1.append(0)
    hash2=[]
    for i in range(dist2.shape[0]):
       if dist2[i] > thre:
           hash2.append(1)
       else:
           hash2.append(0)
    length = hash1.__len__()
    degree1 = (length - np.sum(hash1)) / length * 100
    degree2 = (length - np.sum(hash2)) / length * 100
    return degree1,degree2


def P_Hash(f1,f0, thre=0.005):
    f1 = np.mean(f1, axis=(1, 2))
    f0 = np.mean(f0, axis=(1, 2))
    dist = np.abs(f0-f1)
    hash =[]
    print(thre)
    for i in range(dist.shape[0]):
       if dist[i] > thre:
           hash.append(1)
       else:
           hash.append(0)
    length = hash.__len__()
    degree = (length - np.sum(hash)) / length * 100
    return degree


def Resnet_P_hash(file_fake1, file_fake2, file_real):

    extract_list = ["layer2"]

    Net = models.resnet18(pretrained=True)
    #print(Net)  # 可以打印看模型结构
    if use_gpu:
        Net.cuda()

    image_fake1 = getImage(file_fake1)
    image_fake2 = getImage(file_fake2)
    image_real = getImage(file_real)

    Tensor = torch.cuda.FloatTensor if use_gpu else torch.Tensor
    input1 = Tensor(1, 3, h, w)
    input2 = Tensor(1, 3, h, w)
    input0 = Tensor(1, 3, h, w)

    x1 = torch.unsqueeze(image_fake1, dim=0)
    x2 = torch.unsqueeze(image_fake2, dim=0)
    x0 = torch.unsqueeze(image_real, dim=0)

    fake1_input = Variable(input1.copy_(x1))
    fake2_input = Variable(input2.copy_(x2))
    real_input = Variable(input0.copy_(x0))

    extract_result = FeatureExtractor(Net, extract_list)
    x1_channelfeature = np.squeeze(extract_result(fake1_input)[0].cpu().detach().numpy())
    x2_channelfeature = np.squeeze(extract_result(fake2_input)[0].cpu().detach().numpy())
    x0_channelfeature = np.squeeze(extract_result(real_input)[0].cpu().detach().numpy())

    '''
    degree1,degree2 = getHash(x1_channelfeature,x2_channelfeature,x0_channelfeature, thre=0.005)
    print('our——phash', degree1)
    print('cycle_phash', degree2)
    '''

    degree = P_Hash(x1_channelfeature, x0_channelfeature, thre=0.01)
    print('our——phash', degree)
    degree = P_Hash(x2_channelfeature, x0_channelfeature, thre=0.01)
    print('our——phash', degree)


    print(x1_channelfeature.shape)
    if showfeature:
        plt.figure()
        for i in range(x1_channelfeature.shape[0]):
            plt.subplot(1,3,1)
            plt.imshow(x1_channelfeature[i, :, :])
            plt.subplot(1,3,2)
            plt.imshow(x2_channelfeature[i, :, :])
            plt.subplot(1,3,3)
            plt.imshow(x0_channelfeature[i, :, :])
            plt.waitforbuttonpress()
        plt.show()

if __name__ == '__main__':
    a = 7
    b = 3
    img1 = '/home/zhangbc/Mydataspace/LST/mymodelx288/Cycle_GAN_pathology/epoch3/output/A2B/WSI_data00' + str(a) + '/0' + str(b) + '_pre_B.png'
    img2 = '/home/zhangbc/Mydataspace/LST/mymodelx288/Cycle_GAN/epoch3/output/A2B/WSI_data00' + str(a) + '/0' + str(b) + '_pre_B.png'
    img0 = '/home/zhangbc/Mydataspace/LST/raw_data/NewDataset_x20/Test_aligned/Ki67/WSI_data00' + str(a) + '/0' + str(b) + '.png'
    Resnet_P_hash(img1, img2, img0)

