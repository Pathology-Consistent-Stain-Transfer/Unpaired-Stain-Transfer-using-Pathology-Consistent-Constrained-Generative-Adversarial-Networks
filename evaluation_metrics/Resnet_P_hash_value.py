import torch.nn as nn
from torchvision import models
import os
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pandas import DataFrame
'''
resnet-18 layer3--->thre=0.0025
resnet-18 layer3--->thre=0.005
resnet-18 layer2--->thre=0.01
resnet-18 layer1--->thre=0.02
'''

use_gpu = True
h = 1500
w = 1505

def getImage(filepath, part):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img_data = cv2.imread(filepath)
    if part == 1:
        img_data = img_data[0:1500, 0:1505, :]
    elif part == 2:
        img_data = img_data[0:1500, 1505:3010, :]
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
    #print(thre)
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


def P_Hash(f1, f0, sf_list=None, thre=0.005):
    if sf_list is None:
        f1 = np.mean(f1, axis=(1, 2))
        f0 = np.mean(f0, axis=(1, 2))
    else:
        f1 = np.mean(f1[sf_list, :, :], axis=(1, 2))
        f0 = np.mean(f0[sf_list, :, :], axis=(1, 2))
    dist = np.abs(f0-f1)
    hash =[]
    for i in range(dist.shape[0]):
       if dist[i] > thre:  # if dist[i] > thre:
           hash.append(1)
       else:
           hash.append(0)
    length = hash.__len__()
    degree = (length - np.sum(hash)) / length * 100
    return degree, f1, f0, hash


def Resnet_P_hash2(file_fake1, file_fake2, file_real,showfeature):

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

    degree, _, _, _ = P_Hash(x1_channelfeature, x0_channelfeature, thre=0.01)
    print('our——phash', degree)
    degree, _, _, _ = P_Hash(x2_channelfeature, x0_channelfeature, thre=0.01)
    print('cycle——phash', degree)


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


def Resnet_P_hash(testImage,refImage, part=1, layername="layer1", thre=0.02, sf_list=None, showfeature=False):
    extract_list =[layername]

    Net = models.resnet101(pretrained=True)
    # print(Net)  # 可以打印看模型结构
    if use_gpu:
        Net.cuda()

    image_test = getImage(testImage, part)
    image_ref = getImage(refImage, part)

    Tensor = torch.cuda.FloatTensor if use_gpu else torch.Tensor
    input1 = Tensor(1, 3, h, w)
    input0 = Tensor(1, 3, h, w)

    x1 = torch.unsqueeze(image_test, dim=0)
    x0 = torch.unsqueeze(image_ref, dim=0)

    test_input = Variable(input1.copy_(x1))
    ref_input = Variable(input0.copy_(x0))

    extract_result = FeatureExtractor(Net, extract_list)
    x1_channelfeature = np.squeeze(extract_result(test_input)[0].cpu().detach().numpy())
    x0_channelfeature = np.squeeze(extract_result(ref_input)[0].cpu().detach().numpy())

    degree, f1, f0, hash_tabel = P_Hash(x1_channelfeature, x0_channelfeature, sf_list=sf_list, thre=thre)
    list_A = np.reshape(np.asarray(np.where(np.asarray(hash_tabel)>0)), newshape=[-1])
    # print(list_A.shape)
    # print('list_A', list_A)
    # print('Resnet_phash', degree)
    shape = f1.shape[0]
    f1 = np.reshape(f1, (1, shape))
    f1 = np.tile(f1, (10, 1))
    f0 = np.reshape(f0, (1, shape))
    f0 = np.tile(f0, (10, 1))
    hash_tabel = np.reshape(hash_tabel, (1, shape))
    hash_tabel = np.tile(hash_tabel, (10, 1))
    black = np.zeros(shape=(5, shape))

    show_img = np.vstack((np.vstack((np.vstack((np.vstack((f1, black)), f0)), black)), hash_tabel))
    kk=[9, 12, 18, 27, 28, 29, 38, 45, 47, 64, 85, 103, 111, 126]
    if showfeature:
        plt.figure()
        for idx in range(x1_channelfeature.shape[0]):# x1_channelfeature.shape[0]   list_A.__len__() kk.__len__()
            # i = kk[idx]
            i = idx
            plt.subplot(3, 1, 1)
            plt.imshow(show_img)
            plt.subplot(3,1,2)
            plt.imshow(x1_channelfeature[i, :, :])
            plt.subplot(3,1,3)
            plt.imshow(x0_channelfeature[i, :, :])
            plt.waitforbuttonpress()
            print(i)
        plt.show()
    return degree, list_A

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':

    modelname = ['Cycle_GAN', 'Cycle_GAN_unet', 'Cycle_GAN_UnetSSIM', 'Cycle_GAN_SSIM', 'Cycle_GAN_pathology_cls',
                 'Cycle_GAN_pathology_seg']
    model_id = 5
    print(model_id)
    epoch_id = 7
    # wsi_name = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8]
    # id = [1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 3, 1, 2]
    wsi_name = [4,4,4,4,7,7,7,7,7,7,7,7,7,7, 8,8,8,8,8,8,8,8,8,8, 9,9,9]
    id = [1,2,3,4,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3]
    D_list = np.zeros(shape=(2048))
    xlsdata = {
        'thre': [],
        'p-hash_value_mean': [],
        'p-hash_value_std': []}
    layer_id = 2
    layername = "layer" + str(layer_id)
    for index in range(1, 21):
        # thre = index*0.0005
        thre = index * 0.001
        degree_list = []
        for idx in range(wsi_name.__len__()):
            img1 = '/home/zhangbc/Mydataspace/LST/breast/mymodelx288/' + modelname[model_id] + '/exp2/epoch' + str(
                epoch_id) + '/output/A2B/WSI_data00' + str(wsi_name[idx]) + '/' + str(id[idx]).zfill(2) + '_pre_B.png'
            img0 = '/home/zhangbc/Mydataspace/LST/raw_data/breast/Test/Ki67/WSI_data00' + str(
                wsi_name[idx]) + '/' + str(id[idx]).zfill(2) + '.png'

            degree, A = Resnet_P_hash(img1, img0, part=1, layername=layername, sf_list=None, thre=thre, showfeature=False)
            degree_list.append(degree)
            for i in A:
                D_list[i] += 1

            degree, A = Resnet_P_hash(img1, img0, part=2, layername=layername, sf_list=None, thre=thre, showfeature=False)
            degree_list.append(degree)
            for i in A:
                D_list[i] += 1
        degree_list = np.asarray(degree_list, dtype=np.float32)
        mean_pv = np.mean(degree_list)
        std_pv = np.std(degree_list)
        print(index)
        print('Resnet_phash', mean_pv)
        print('Resnet_phash_std', std_pv)
        xlsdata['thre'].append(thre)
        xlsdata['p-hash_value_mean'].append(mean_pv)
        xlsdata['p-hash_value_std'].append(std_pv)

    df = DataFrame(xlsdata)
    order = ['thre', 'p-hash_value_mean', 'p-hash_value_std']
    df = df[order]
    df.to_excel('/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/' + modelname[model_id] + '/epoch' + str(
        epoch_id) + '/output/B_B_pv_'+layername+'.xlsx')

'''
    print(D_list)
    d_flist = np.where(D_list>0)
    print(d_flist)
    DF = DF + np.reshape(np.asarray(d_flist), [-1]).tolist()
    print(np.unique(DF).tolist())


    DF =[15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 120, 122, 123, 125, 126, 127, 129, 130, 131, 132, 133, 134, 136, 137, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 174, 175, 176, 179, 180, 181, 184, 187, 188, 189, 192, 194, 195, 196, 197, 198, 201, 202, 203, 205, 207, 208, 209, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 235, 236, 237, 238, 239, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 258, 259, 260, 261, 264, 265, 267, 271, 273, 275, 276, 277, 278, 279, 280, 282, 283, 284, 285, 287, 288, 289, 291, 292, 293, 294, 295, 296, 299, 301, 302, 303, 304, 307, 308, 310, 311, 315, 316, 317, 318, 319, 321, 322, 323, 324, 325, 326, 327, 328, 330, 331, 332, 334, 335, 336, 337, 339, 346, 348, 350, 352, 354, 356, 357, 358, 359, 362, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 376, 377, 378, 379, 384, 386, 387, 390, 391, 392, 393, 395, 396, 397, 398, 399, 401, 402, 403, 404, 407, 410, 411, 412, 414, 415, 416, 417, 418, 420, 421, 422, 423, 424, 425, 427, 428, 430, 431, 432, 434, 435, 438, 439, 442, 443, 444, 445, 446, 447, 448, 449, 450, 452, 453, 454, 455, 456, 457, 458, 459, 460, 464, 465, 466, 467, 468, 469, 470, 472, 473, 475, 476, 477, 478, 479, 482, 483, 484, 485, 486, 487, 488, 489, 490, 494, 496, 497, 498, 499, 500, 502, 503, 504, 505, 509, 510, 511]
        
'''
