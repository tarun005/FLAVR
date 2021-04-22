import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob


class UCF(Dataset):
    def __init__(self, data_root , ext="png"):

        super().__init__()

        self.data_root = data_root
        self.file_list = sorted(os.listdir(self.data_root))

        self.transforms = transforms.Compose([
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, idx):

        imgpath = os.path.join(self.data_root , self.file_list[idx])
        imgpaths = [os.path.join(imgpath , "frame0.png") , os.path.join(imgpath , "frame1.png") ,os.path.join(imgpath , "frame2.png") ,os.path.join(imgpath , "frame3.png") ,os.path.join(imgpath , "framet.png")]

        images = [Image.open(img) for img in imgpaths]
        images = [self.transforms(img) for img in images]

        return images[:-1] , [images[-1]]

    def __len__(self):

        return len(self.file_list)

def get_loader(data_root, batch_size, shuffle, num_workers, test_mode=True):

    dataset = UCF(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataset = UCF_triplet("./ucf_test/")

    print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)
