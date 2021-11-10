#!/usr/bin/python3

import argparse
import itertools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch

from mymodels import Generator_resnet
from mymodels import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from utils import MS_SSIM_Loss
from datasets import ImageDataset
# breast/ neuroendocrine / GLAS
parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', type=str, default='/home/zhangbc/Mydataspace/LST/breast/datasetX20_288/Train', help='root directory of the dataset')
parser.add_argument('--modelroot', type=str, default='/home/zhangbc/Mydataspace/LST/breast/mymodelx288/Cycle_GAN_SSIM', help='root directory of the model')

parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=12, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=2, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=288, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')

parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

parser.add_argument('--continue_train', type=bool, default=False, help='load model and continue trainning')
parser.add_argument('--loadroot', type=str, default='/home/zhangbc/Mydataspace/LST/breast/mymodelx288/Cycle_GAN_SSIM/temp', help='continue train root directory of the model')
opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator_resnet(opt.input_nc, opt.output_nc, 10, False)
netG_B2A = Generator_resnet(opt.output_nc, opt.input_nc, 10, False)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
criterion_ssim = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

transforms_ = transforms.Compose([
    transforms.RandomCrop(opt.size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, batch_size=opt.batchSize, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

###################################
start_epoch = opt.epoch

if opt.continue_train:

    netG_A2B_checkpoint = torch.load(os.path.join(opt.loadroot, 'netG_A2B.pth'))  # 加载断点
    netG_A2B.load_state_dict(netG_A2B_checkpoint['model'])  # 加载模型可学习参数

    netG_B2A_checkpoint = torch.load(os.path.join(opt.loadroot, 'netG_B2A.pth'))  # 加载断点
    netG_B2A.load_state_dict(netG_B2A_checkpoint['model'])  # 加载模型可学习参数
    optimizer_G.load_state_dict(netG_B2A_checkpoint['optimizer'])  # 加载优化器参数
    lr_scheduler_G.load_state_dict(netG_B2A_checkpoint['lr_schedule'])  # 加载lr_scheduler
    start_epoch = netG_B2A_checkpoint['epoch']

    netD_A_checkpoint = torch.load(os.path.join(opt.loadroot, 'netD_A.pth'))  # 加载断点
    netD_A.load_state_dict(netD_A_checkpoint['model'])  # 加载模型可学习参数
    optimizer_D_A.load_state_dict(netD_A_checkpoint['optimizer'])  # 加载优化器参数
    lr_scheduler_D_A.load_state_dict(netD_A_checkpoint['lr_schedule'])  # 加载lr_scheduler

    netD_B_checkpoint = torch.load(os.path.join(opt.loadroot, 'netD_B.pth'))  # 加载断点
    netD_B.load_state_dict(netD_B_checkpoint['model'])  # 加载模型可学习参数
    optimizer_D_B.load_state_dict(netD_B_checkpoint['optimizer'])  # 加载优化器参数
    lr_scheduler_D_B.load_state_dict(netD_B_checkpoint['lr_schedule'])  # 加载lr_scheduler

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), start_epoch)

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(input_A.copy_(batch['HE']))
        real_B = Variable(input_B.copy_(batch['Ki67']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # TUM remove the Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B, _ = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)
        # G_B2A(A) should equal A if real A is fed
        same_A, _ = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)

        # GAN loss
        fake_B, features_fb = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A, features_fa = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A, features_ra = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
        criterion_ssimABA = criterion_ssim(recovered_A, real_A)

        recovered_B, features_rb = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)
        criterion_ssimBAB = criterion_ssim(recovered_B, real_B)

        # Total loss
        loss_G = 5.0 * (loss_identity_A + loss_identity_B) + \
                 1.0 * (loss_GAN_A2B + loss_GAN_B2A) + \
                 10.0 * (loss_cycle_ABA + loss_cycle_BAB) + \
                 2.0 * (criterion_ssimABA + criterion_ssimBAB)

        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_Ad = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_Ad.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_Bd = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_Bd.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G,
                    'loss_G_SSIM': (criterion_ssimABA + criterion_ssimBAB),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                    'loss_D': (loss_D_A + loss_D_B)},
                   images={'real_cycleSSIM_A': real_A, 'real_cycleSSIM_B': real_B,
                           'fake_cycleSSIM_A': fake_A, 'fake_cycleSSIM_B': fake_B})

        # save models at half of an epoch
        if (i + 1) % (dataloader.__len__() // 5 + 1) == 0:
            saveroot = os.path.join(opt.modelroot, 'temp')
            if not os.path.exists(saveroot):
                os.makedirs(saveroot)

            # Save models checkpoints
            netG_A2B_checkpoints = {
                "model": netG_A2B.state_dict()
            }
            torch.save(netG_A2B_checkpoints, os.path.join(saveroot, 'netG_A2B.pth'))

            netG_B2A_checkpoints = {
                "model": netG_B2A.state_dict(),
                'optimizer': optimizer_G.state_dict(),
                "epoch": epoch,
                'lr_schedule': lr_scheduler_G.state_dict()
            }
            torch.save(netG_B2A_checkpoints, os.path.join(saveroot, 'netG_B2A.pth'))

            netD_A_checkpoints = {
                "model": netD_A.state_dict(),
                'optimizer': optimizer_D_A.state_dict(),
                'lr_schedule': lr_scheduler_D_A.state_dict()
            }
            torch.save(netD_A_checkpoints, os.path.join(saveroot, 'netD_A.pth'))

            netD_B_checkpoints = {
                "model": netD_B.state_dict(),
                'optimizer': optimizer_D_B.state_dict(),
                'lr_schedule': lr_scheduler_D_B.state_dict()
            }
            torch.save(netD_B_checkpoints, os.path.join(saveroot, 'netD_B.pth'))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    saveroot = os.path.join(opt.modelroot, 'epoch'+str(epoch))
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), os.path.join(saveroot, 'netG_A2B.pth'))
    torch.save(netG_B2A.state_dict(), os.path.join(saveroot, 'netG_B2A.pth'))
    torch.save(netD_A.state_dict(), os.path.join(saveroot, 'netD_A.pth'))
    torch.save(netD_B.state_dict(), os.path.join(saveroot, 'netD_B.pth'))
