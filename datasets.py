import glob
import random
import os
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, batch_size=None, unaligned=False):
        self.transform = transforms_
        self.unaligned = unaligned
        self.batch_size = batch_size
        self.files_A = sorted(glob.glob(os.path.join(root, 'HE') + '/*/*.png'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'Ki67') + '/*/*.png'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        # print(self.files_A[index % len(self.files_A)], '\n')
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'HE': item_A, 'Ki67': item_B}

    def __len__(self):
        return max(len(self.files_A)//self.batch_size * self.batch_size,
                   len(self.files_B)//self.batch_size * self.batch_size)


class ExpertDataset_label(Dataset):
    def __init__(self, export_root, transform_expert=None, batch_size=None, unaligned=False):
        self.transform_expert = transform_expert
        self.batch_size = batch_size
        self.unaligned = unaligned
        self.files_expert_A = sorted(glob.glob(os.path.join(export_root, 'HE') + '/*/*.npz'))
        self.files_expert_B = sorted(glob.glob(os.path.join(export_root, 'Ki67') + '/*/*.npz'))

    def __getitem__(self, index):

        expert_data_A = np.load(self.files_expert_A[index % len(self.files_expert_A)])
        expert_item_A = expert_data_A['image']
        expert_item_A = Image.fromarray(expert_item_A.astype('uint8')).convert('RGB')
        expert_item_A = self.transform_expert(expert_item_A)
        expert_item_A_p = torch.from_numpy(np.asarray(expert_data_A['proportion'], dtype=np.float32))
        expert_item_A_l = torch.from_numpy(np.asarray(expert_data_A['label'], dtype=np.float32))

        if self.unaligned:
            expert_data_B = np.load(self.files_expert_B[random.randint(0, len(self.files_expert_B) - 1)])
        else:
            expert_data_B = np.load(self.files_expert_B[index % len(self.files_expert_B)])

        expert_item_B = expert_data_B['image']
        expert_item_B = Image.fromarray(expert_item_B.astype('uint8')).convert('RGB')
        expert_item_B = self.transform_expert(expert_item_B)
        expert_item_B_p = torch.from_numpy(np.asarray(expert_data_B['proportion'], dtype=np.float32))
        expert_item_B_l = torch.from_numpy(np.asarray(expert_data_B['label'], dtype=np.float32))

        return {'expert_HE': expert_item_A, 'expert_Ki67': expert_item_B,
                'expert_HE_label': expert_item_A_l, 'expert_Ki67_label': expert_item_B_l,
                'expert_HE_p': expert_item_A_p, 'expert_Ki67_p': expert_item_B_p, }

    def __len__(self):
        return max(len(self.files_expert_A)//self.batch_size * self.batch_size,
                   len(self.files_expert_B)//self.batch_size * self.batch_size)


class ExpertDataset_mask(Dataset):
    def __init__(self, export_root, batch_size=None, unaligned=False):
        self.batch_size = batch_size
        self.transform_img = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.transform_mask = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()])
        self.unaligned = unaligned
        self.files_expert_A = sorted(glob.glob(os.path.join(export_root, 'HE') + '/*/*.npz'))
        self.files_expert_B = sorted(glob.glob(os.path.join(export_root, 'Ki67') + '/*/*.npz'))

    def __getitem__(self, index):

        expert_data_A = np.load(self.files_expert_A[index % len(self.files_expert_A)])
        expert_item_A = expert_data_A['image']
        expert_item_A = Image.fromarray(expert_item_A.astype('uint8')).convert('RGB')

        expert_item_A_mask = expert_data_A['mask']*255.0
        expert_item_A_mask = Image.fromarray(expert_item_A_mask.astype('uint8'))

        if self.unaligned:
            expert_data_B = np.load(self.files_expert_B[random.randint(0, len(self.files_expert_B) - 1)])
        else:
            expert_data_B = np.load(self.files_expert_B[index % len(self.files_expert_B)])

        seed = np.random.randint(2147483647)
        random.seed(seed)  # apply this seed to img tranfsorms
        expert_item_A = self.transform_img(expert_item_A)

        random.seed(seed)  # apply this seed to target tranfsorms
        expert_item_A_mask = self.transform_mask(expert_item_A_mask)

        expert_item_B = expert_data_B['image']
        expert_item_B = Image.fromarray(expert_item_B.astype('uint8')).convert('RGB')

        expert_item_B_mask = expert_data_B['mask']*255.0
        expert_item_B_mask = Image.fromarray(expert_item_B_mask.astype('uint8'))

        seed = np.random.randint(2147483648)
        random.seed(seed)  # apply this seed to img tranfsorms
        expert_item_B = self.transform_img(expert_item_B)

        random.seed(seed)  # apply this seed to target tranfsorms
        expert_item_B_mask = self.transform_mask(expert_item_B_mask)

        return {'expert_HE': expert_item_A, 'expert_Ki67': expert_item_B,
                'expert_HE_mask': expert_item_A_mask, 'expert_Ki67_mask': expert_item_B_mask}

    def __len__(self):
        return max(len(self.files_expert_A)//self.batch_size * self.batch_size,
                   len(self.files_expert_B)//self.batch_size * self.batch_size)