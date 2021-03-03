## Imported from https://github.com/myungsub/CAIN/blob/master/data/snufilm.py
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SNUFILM(Dataset):
    def __init__(self, data_root, mode='hard'):
        '''
        :param data_root:   ./data/SNU-FILM
        :param mode:        ['easy', 'medium', 'hard', 'extreme']
        '''
        test_root = os.path.join(data_root, 'test')
        test_fn = os.path.join(data_root, 'test-%s.txt' % mode)
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()

        self.root = data_root

        if mode == "easy":
            nbrs = [-3,-1,0,1,3]
        elif mode == "medium":
            nbrs = [-6,-2,0,2,6]
        elif mode == "hard":
            nbrs = [-12,-4,0,4,12]
        elif mode == "extreme":
            nbrs = [-24,-8,0,8,24]

        self.input_frame_list = []
        for v in self.frame_list:
            files = v.split(' ')[1].split('/')
            folder = "/".join(files[:-1])
            frame_id = files[-1]
            span = len(frame_id.partition(".png")[0])
            frame_no = int(frame_id.partition(".png")[0])
            if frame_no + nbrs[0] <0 or frame_no + nbrs[1] < 0:
                continue; ## Invalid input frames
            if span==6:
                input_frame = ["%06d.png"%(frame_no+i) for i in nbrs]
            elif span==5:
                input_frame = ["%05d.png"%(frame_no+i) for i in nbrs]

            if not all([os.path.exists(os.path.join(self.root , folder , tf)) for tf in input_frame]):
                continue;

            frame_ids = [os.path.join(folder , inp_) for inp_ in input_frame]
            self.input_frame_list.append(frame_ids)
        
        self.transforms = transforms.Compose([
            transforms.CenterCrop(720),
            transforms.ToTensor()
        ])
        
        print("[%s] Test dataset has %d quadrapulets" %  (mode, len(self.input_frame_list)))


    def __getitem__(self, index):
        
        imgpaths = self.input_frame_list[index]

        images = [Image.open(os.path.join(self.root , imgpath_)) for imgpath_ in imgpaths]
        images = [self.transforms(img_) for img_ in images]

        frameRange = [0,1,3,4]
        gt_image = images[2]
        frames = [images[idx_] for idx_ in frameRange]

        return frames, gt_image

    def __len__(self):
        return len(self.input_frame_list)


def get_loader(test_mode , data_root, batch_size, shuffle, num_workers=0):
    dataset = SNUFILM(data_root, mode=test_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataloader = get_loader(test_mode="easy" , data_root="./snu_test/" , batch_size=12 , shuffle=True)
