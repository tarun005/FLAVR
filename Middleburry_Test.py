import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.transforms import Resize

import config
import myutils

from torch.utils.data import DataLoader


args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

from dataset.Middleburry import get_loader
test_loader = get_loader(args.data_root, 1, shuffle=False, num_workers=args.num_workers)   


from model.FLAVR_arch import UNet_3D_3D
print("Building model: %s"%args.model.lower())
model = UNet_3D_3D(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType)


# Just make every model to DataParallel
model = torch.nn.DataParallel(model).to(device)
print("#params" , sum([p.numel() for p in model.parameters()]))

def make_image(img):
    # img = F.interpolate(img.unsqueeze(0) , (720,1280) , mode="bilinear").squeeze(0)
    q_im = img.data.mul(255.).clamp(0,255).round()
    im = q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return im

folderList = ['Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Mequon', 'Schefflera', 'Teddy', 'Urban']

def test(args):
    time_taken = []
    img_save_id = 0
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    psnr_list = []
    with torch.no_grad():
        for i, (images, name ) in enumerate((test_loader)):

            if name[0] not in folderList:
                continue;

            images = torch.stack(images , dim=1).squeeze(0)

            # images = [img_.cuda() for img_ in images]

            H,W = images[0].shape[-2:]
            resizes = 8*(H//8) , 8*(W//8)

            import torchvision
            transform = Resize(resizes)
            rev_transforms = Resize((H,W))
            images = transform(images).unsqueeze(0).cuda()# [transform(img_.squeeze(0)).unsqueeze(0).cuda() for img_ in images]
            images = torch.unbind(images, dim=1)

            start_time = time.time()
            out = model(images)
            print("Time Taken" , time.time() - start_time)

            out = torch.cat(out)
            out = rev_transforms(out)
            
            output_image = make_image(out.squeeze(0))

            import imageio
            os.makedirs("Middleburry/%s/"%name[0])
            imageio.imwrite("Middleburry/%s/frame10i11.png"%name[0], output_image) 

    
    return

def main(args):
    
    assert args.load_from is not None

    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
