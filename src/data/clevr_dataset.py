import json
import os
import pickle
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import torch
from torchvision import transforms


class ClevrDatasetImages(Dataset):
    """
    Loads only images from the CLEVR dataset
    """

    def __init__(self, clevr_dir, train, transform=None):
        """
        :param clevr_dir: Root directory of CLEVR dataset
        :param mode: Specifies if we want to read in val, train or test folder
        :param transform: Optional transform to be applied on a sample.
        """
        self.mode = 'train' if train else 'val'
        
        self.img_dir = os.path.join(clevr_dir, 'images', self.mode)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        padded_index = str(idx).rjust(6, '0')
        img_filename = os.path.join(self.img_dir, 'CLEVR_{}_{}.png'.format(self.mode,padded_index))
        image = Image.open(img_filename).convert('RGB')
        image = transforms.functional.crop(image, 64, 29, 192, 192)
        if self.transform:
            image = self.transform(image)

        return image

def return_data(clevr_dir, batch_size, num_workers=1):

    train_transforms = transforms.Compose([transforms.Resize((64, 64)),
                                        #transforms.RandomRotation(2.8),  # .05 rad
                                        transforms.ToTensor()])
    #test_transforms = transforms.Compose([transforms.Resize((128, 128)),
    #                                    transforms.ToTensor()])
                                        
    clevr_dataset_train = ClevrDatasetImages(clevr_dir, True, train_transforms)
    #clevr_dataset_test = ClevrDatasetImages(clevr_dir, False, test_transforms)
    train_loader = DataLoader(clevr_dataset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    return train_loader