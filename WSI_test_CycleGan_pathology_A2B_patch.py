#!/usr/bin/python3

import argparse
import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from mymodels import Generator_unet_seg, Generator_unet_cls
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestImage(Dataset):
    def __init__(self, Image_data, pos_list,transforms_=None, patch_size=512):
        self.transform = transforms_
        self.Image = Image_data
        self.pos_list = pos_list
        self.patch_size = patch_size

    def __getitem__(self, index):
        item = self.transform(self.Image[self.pos_list[index][0] - self.patch_size:self.pos_list[index][0],
                                 self.pos_list[index][1] - self.patch_size:self.pos_list[index][1], :])

        return {'A': item, 'x': self.pos_list[index][0], 'y': self.pos_list[index][1]}

    def __len__(self):
        return len(self.pos_list)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--filepath', type=str, help='root directory of the testimage')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=576, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
opt = parser.parse_args()
# breast 5 / neuroendocrine 7 / GLAS
datasetname = 'neuroendocrine'

# opt.generator_A2B = '/home/zhangbc/Mydataspace/LST/'+datasetname+'/mymodelx288/Cycle_GAN_pathology_seg/exp1/epoch3/netG_A2B.pth'
# opt.generator_B2A = '/home/zhangbc/Mydataspace/LST/'+datasetname+'/mymodelx288/Cycle_GAN_pathology_seg/exp1/epoch3/netG_B2A.pth'
opt.generator_A2B = '/home/zhangbc/Mydataspace/LST/'+datasetname+'/mymodelx288/Cycle_GAN_pathology_seg_PN3_1/other_epoch/epoch9/netG_A2B.pth'
opt.generator_B2A = '/home/zhangbc/Mydataspace/LST/'+datasetname+'/mymodelx288/Cycle_GAN_pathology_seg_PN3_1/other_epoch/epoch9/netG_B2A.pth'
print(opt)
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator_unet_seg(opt.input_nc, opt.output_nc, 10, alt_leak=True, neg_slope=0.1)
netG_B2A = Generator_unet_seg(opt.output_nc, opt.input_nc, 10, alt_leak=True, neg_slope=0.1)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

# neuroendocrine
# wsi_name = [1,1,2,2,2,3,3,4,4,4,4,4,5,5,6,6,7,7,7,8,8]
# id = [1,2,1,2,3,1,2,1,2,3,4,5,1,2,1,2,1,2,3,1,2]
wsi_name = [1,1,2,2,2,3,3,4,4,4,4,4,5,5,6,6,7,7,7,8,8]
id = [1,2,1,2,3,1,2,1,2,3,4,5,1,2,1,2,1,2,3,1,2]
# breast
# wsi_name = [4,4,4,4,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8, 9,9,9]
# id = [1,2,3,4,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3]

for i in range(wsi_name.__len__()):
    print(i)
    opt.filepath = '/home/zhangbc/Desktop/Mydataspace/LST/raw_data/'+datasetname+'/Test/HE/WSI_data00'+str(wsi_name[i])+'/'+str(id[i]).zfill(2)+'.png'
    if not os.path.exists(opt.filepath):
        print('wrong filepath')
        exit(0)
    # read test data and add padding
    Testimg_data = cv2.imread(opt.filepath)
    Testimg_data = cv2.copyMakeBorder(Testimg_data, opt.size//4, opt.size//4, opt.size//4, opt.size//4, cv2.BORDER_REFLECT)
    Testimg_data = np.array(cv2.cvtColor(Testimg_data, cv2.COLOR_BGR2RGB)).astype(np.uint8)

    filename = opt.filepath.split('/')[-1].split('.')[0]
    Testimg_shape = Testimg_data.shape
    X_index = list(range(opt.size, Testimg_shape[0], int(opt.size / 2)))
    X_index.append(Testimg_shape[0])
    X_index = np.unique(np.asarray(X_index, dtype=np.int))
    Y_index = list(range(opt.size, Testimg_shape[1], int(opt.size / 2)))
    Y_index.append(Testimg_shape[1])
    Y_index = np.unique(np.asarray(Y_index, dtype=np.int))
    pos_list = []
    for x_id in X_index:
        for y_id in Y_index:
            pos_list.append([x_id, y_id])
    Testimg_data = Testimg_data[0:X_index[-1], 0:Y_index[-1], :]
    Pre_img = np.zeros_like(Testimg_data)
    rec_img = np.zeros_like(Testimg_data)
    # Dataset loader
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataloader = DataLoader(TestImage(Testimg_data, pos_list, transforms_=transforms_, patch_size=opt.size),
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    for i, batch in enumerate(dataloader):
        # Set model input
        x_ = batch['x'].numpy()  # list of x position
        y_ = batch['y'].numpy()
        real_A = Variable(input_A.copy_(batch['A']))
        # Generate output
        fake_B, _, c_out, _ = netG_A2B(real_A)
        rec_A, _, _, _ = netG_B2A(fake_B)
        fake_B = 0.5 * (fake_B.data + 1.0)
        rec_A = 0.5 * (rec_A.data + 1.0)
        #print(c_out.data)

        for patch_id in range(opt.batchSize):
            pre_patch_B = fake_B[patch_id].squeeze()
            pre_patch_B_np = torch.mul(pre_patch_B, 255).__add__(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                         torch.uint8).numpy()

            pre_patch_B_np_center = pre_patch_B_np[int(opt.size / 4):int(opt.size / 4) * 3,
                                    int(opt.size / 4):int(opt.size / 4) * 3, :]
            # Save image files
            Pre_img[x_[patch_id] - opt.size + int(opt.size / 4):x_[patch_id] - opt.size + int(opt.size / 4) * 3,
            y_[patch_id] - opt.size + int(opt.size / 4):y_[patch_id] - opt.size + int(opt.size / 4) * 3,
            :] = pre_patch_B_np_center

            pre_patch_A = rec_A[patch_id].squeeze()
            pre_patch_A_np = torch.mul(pre_patch_A, 255).__add__(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                         torch.uint8).numpy()

            pre_patch_A_np_center = pre_patch_A_np[int(opt.size / 4):int(opt.size / 4) * 3,
                                    int(opt.size / 4):int(opt.size / 4) * 3, :]
            # Save image files
            rec_img[x_[patch_id] - opt.size + int(opt.size / 4):x_[patch_id] - opt.size + int(opt.size / 4) * 3,
            y_[patch_id] - opt.size + int(opt.size / 4):y_[patch_id] - opt.size + int(opt.size / 4) * 3,
            :] = pre_patch_A_np_center

        sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

    Pre_img = Pre_img[int(opt.size / 4):X_index[-1] - int(opt.size / 4),
              int(opt.size / 4):Y_index[-1] - int(opt.size / 4), :]
    rec_img = rec_img[int(opt.size / 4):X_index[-1] - int(opt.size / 4),
              int(opt.size / 4):Y_index[-1] - int(opt.size / 4), :]
    Testimg_data = Testimg_data[int(opt.size / 4):X_index[-1] - int(opt.size / 4),
                   int(opt.size / 4):Y_index[-1] - int(opt.size / 4), :]
    Pre_img = Image.fromarray(Pre_img)
    rec_img = Image.fromarray(rec_img)
    Testimg = Image.fromarray(Testimg_data)
    save_dir = os.path.join(os.path.dirname(opt.generator_A2B), 'output/A2B/')
    save_dir = os.path.join(save_dir, os.path.dirname(opt.filepath).split('/')[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Pre_img.save(os.path.join(save_dir, filename + '_pre_B.png'))
    Testimg.save(os.path.join(save_dir, filename + '_real_A.png'))
    rec_img.save(os.path.join(save_dir, filename + '_rec_A.png'))
    sys.stdout.write('\n')