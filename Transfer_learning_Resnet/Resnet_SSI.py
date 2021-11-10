import torch.nn as nn
from torchvision import models, transforms
import argparse
import sys
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TestImage(Dataset):
    def __init__(self, filepath, patch_size=256):
        self.Image, self.pos_list = self.getImage(filepath)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.patch_size = patch_size

    def __getitem__(self, index):
        item = self.transform(self.Image[self.pos_list[index][0] - self.patch_size:self.pos_list[index][0],
                                 self.pos_list[index][1] - self.patch_size:self.pos_list[index][1], :])

        return {'Img': item}

    def __len__(self):
        return len(self.pos_list)

    def getImage(self, filepath):
        Testimg_data = np.array(Image.open(filepath).convert('RGB')).astype(np.uint8)
        Testimg_shape = Testimg_data.shape
        X_index = list(range(opt.size, Testimg_shape[0], int(opt.size)))
        X_index.append(Testimg_shape[0])
        X_index = np.unique(np.asarray(X_index, dtype=np.int))
        Y_index = list(range(opt.size, Testimg_shape[1], int(opt.size)))
        Y_index.append(Testimg_shape[1])
        Y_index = np.unique(np.asarray(Y_index, dtype=np.int))
        pos_list = []
        for x_id in X_index:
            for y_id in Y_index:
                pos_list.append([x_id, y_id])
        return Testimg_data, pos_list


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

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=150, help='size of the data (squared assumed)')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)


def Resnet_SSI(file_fake, file_real, file_real2):

    extract_list = ["avgpool"]

    resnet = models.resnet18(pretrained=True)
    print(resnet)  # 可以打印看模型结构
    if opt.cuda:
        resnet.cuda()

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input1 = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size*2)
    input2 = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size*2)
    input3 = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size*2)

    dataloader_fake_IHC = DataLoader(TestImage(file_fake, patch_size=opt.size),
                                 batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    dataloader_real_HE = DataLoader(TestImage(file_real, patch_size=opt.size),
                                 batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    dataloader_real_IHC = DataLoader(TestImage(file_real2, patch_size=opt.size),
                                 batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    MSSI_fakeIHCImg_vector=[]
    for i, batch in enumerate(dataloader_fake_IHC):
        # Set model input
        fake_input = Variable(input1.copy_(batch['Img']))
        extract_result = FeatureExtractor(resnet, extract_list)
        SSI_fake_vector = extract_result(fake_input)[0].cpu().detach().numpy()
        for patch_id in range(opt.batchSize):
            MSSI_fakeIHCImg_vector.append(SSI_fake_vector[patch_id].squeeze())
    MSSI_fakeIHCImg_vector = np.squeeze(np.mean(np.array(MSSI_fakeIHCImg_vector), axis=0))

    MSSI_realHEImg_vector = []
    for i, batch in enumerate(dataloader_real_HE):
        # Set model input
        real_input = Variable(input2.copy_(batch['Img']))
        extract_result = FeatureExtractor(resnet, extract_list)
        SSI_real_vector = extract_result(real_input)[0].cpu().detach().numpy()
        for patch_id in range(opt.batchSize):
            MSSI_realHEImg_vector.append(SSI_real_vector[patch_id].squeeze())

    MSSI_realHEImg_vector = np.squeeze(np.mean(np.array(MSSI_realHEImg_vector), axis=0))

    MSSI_realIHCImg_vector = []
    for i, batch in enumerate(dataloader_real_IHC):
        # Set model input
        real_input2 = Variable(input3.copy_(batch['Img']))
        extract_result = FeatureExtractor(resnet, extract_list)
        SSI_real_vector2 = extract_result(real_input2)[0].cpu().detach().numpy()
        for patch_id in range(opt.batchSize):
            MSSI_realIHCImg_vector.append(SSI_real_vector2[patch_id].squeeze())
    MSSI_realIHCImg_vector = np.squeeze(np.mean(np.array(MSSI_realIHCImg_vector), axis=0))

    Vec_HE2fakeIHC = (MSSI_realHEImg_vector - MSSI_fakeIHCImg_vector)*len(MSSI_realHEImg_vector)
    Vec_HE2realIHC = (MSSI_realHEImg_vector - MSSI_realIHCImg_vector)*len(MSSI_realHEImg_vector)
    Vec_fakeIHC2realIHC = (MSSI_fakeIHCImg_vector - MSSI_realIHCImg_vector)*len(MSSI_realHEImg_vector)

    D = 1-np.linalg.norm(Vec_fakeIHC2realIHC)/np.linalg.norm(Vec_HE2realIHC)

    Vec_HE2fakeIHC = Vec_HE2fakeIHC / np.linalg.norm(Vec_HE2fakeIHC)
    Vec_HE2realIHC = Vec_HE2realIHC / np.linalg.norm(Vec_HE2realIHC)

    MSSI = ((np.dot(Vec_HE2fakeIHC, Vec_HE2realIHC)+1)/2+D)/2
           #/np.maximum(np.linalg.norm(Vec_HE2fakeIHC), np.linalg.norm(Vec_HE2realIHC))

    return MSSI

if __name__ == '__main__':
    file_fake = '/home/zhangbc/Mydataspace/LST/mymodel/HE2IHC_20/epoch19/output/A2B/H6_pre_B.png'
    file_real2 = '/home/zhangbc/Mydataspace/LST/raw_data/HE_IHC_20/test/K6.png'
    file_real = '/home/zhangbc/Mydataspace/LST/raw_data/HE_IHC_20/test/H6.png'
    MSSI = Resnet_SSI(file_fake, file_real, file_real2)  # b-B:c+E
    print("MSSI:", MSSI)
