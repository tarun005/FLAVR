import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob
import pdb


class Adobe240_train(Dataset):
    def __init__(self, data_root , mode, ext="png"):

        super().__init__()

        if mode == "test":
            mode = "validation"

        mode = "train"

        self.data_root = os.path.join(data_root , mode) 
        self.file_list = sorted(os.listdir(self.data_root))

        self.transforms = transforms.Compose([
                # transforms.CenterCrop((360,480)),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                # transforms.Normalize([.5]*3 , [1.]*3)
            ])

    def __getitem__(self, idx):

        imgpath = os.path.join(self.data_root , self.file_list[idx])
        
        imgpaths = [os.path.join(imgpath , fn) for fn in sorted(os.listdir(imgpath))]
        assert len(imgpaths) == 25

        # if random.random() > 0.5:
        #     imgpaths = imgpaths[::-1]

        pick_idxs = [0,8,16,24]
        gt_idxs = list(range(9,16))

        image_paths = [imgpaths[idx_] for idx_ in pick_idxs]
        images = [Image.open(pth) for pth in image_paths]
        images = [self.transforms(img) for img in images]

        gt_paths = [imgpaths[idx_] for idx_ in gt_idxs]
        gt_images = [Image.open(pth) for pth in gt_paths]
        gt_images = [self.transforms(gt_img_) for gt_img_ in gt_images]

        return images , gt_images

    def __len__(self):

        return len(self.file_list)

def get_loader(data_root, mode, batch_size, shuffle, num_workers):

    dataset = Adobe240_train(data_root , mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataset = Adobe240_train("/home/tarunk/FAIR_TSR/datasets/adobe/" , mode="train")

    print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)