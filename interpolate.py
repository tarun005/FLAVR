import os
import torch
import cv2
import pdb
import time
import sys

import torchvision
from PIL import Image
import numpy as np
import tqdm
from torchvision.io import read_video , write_video
from dataset.transforms import ToTensorVideo , Resize

import torch.nn.functional as F


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_video" , type=str , required=True , help="Path/WebURL to input video")
parser.add_argument("--youtube-dl" , type=str , help="Path to youtube_dl" , default=".local/bin/youtube-dl")
parser.add_argument("--factor" , type=int , required=True , choices=[2,4,8] , help="How much interpolation needed. 2x/4x/8x.")
parser.add_argument("--codec" , type=str , help="video codec" , default="mpeg4")
parser.add_argument("--load_model" , required=True , type=str , help="path for stored model")
parser.add_argument("--up_mode" , type=str , help="Upsample Mode" , default="transpose")
parser.add_argument("--output_ext" , type=str , help="Output video format" , default=".avi")
parser.add_argument("--input_ext" , type=str, help="Input video format", default=".mp4")
parser.add_argument("--downscale" , type=float , help="Downscale input res. for memory" , default=1)
parser.add_argument("--output_fps" , type=int , help="Target FPS" , default=30)
parser.add_argument("--is_folder" , action="store_true" )
args = parser.parse_args()

input_video = args.input_video
input_ext = args.input_ext

from os import path

if not args.is_folder and not path.exists(input_video):
    print("Invalid input file path!")
    exit()
    
if args.is_folder and not path.exists(input_video):
    print("Invalid input directory path!")
    exit()

if args.output_ext != ".avi":
    print("Currently supporting only writing to avi. Try using ffmpeg for conversion to mp4 etc.")

if input_video.endswith("/"):
    video_name = input_video.split("/")[-2].split(input_ext)[0]
else:
    video_name = input_video.split("/")[-1].split(input_ext)[0]

output_video = os.path.join(video_name + f"_{args.factor}x" + str(args.output_ext))

n_outputs = args.factor - 1

model_name = "unet_18"
nbr_frame = 4
joinType = "concat"

if input_video.startswith("http"):
    assert args.youtube_dl is not None
    youtube_dl_path = args.youtube_dl
    cmd = f"{youtube_dl_path} -i -o video.mp4 {input_video}"
    os.system(cmd)
    input_video = "video.mp4"
    output_video = "video" + str(args.output_ext) 

def loadModel(model, checkpoint):
    
    saved_state_dict = torch.load(checkpoint)['state_dict']
    saved_state_dict = {k.partition("module.")[-1]:v for k,v in saved_state_dict.items()}
    model.load_state_dict(saved_state_dict)

checkpoint = args.load_model
from model.FLAVR_arch import UNet_3D_3D

model = UNet_3D_3D(model_name.lower() , n_inputs=4, n_outputs=n_outputs,  joinType=joinType , upmode=args.up_mode)
loadModel(model , checkpoint)
model = model.cuda()

def write_video_cv2(frames , video_name , fps , sizes):

    out = cv2.VideoWriter(video_name,cv2.CAP_OPENCV_MJPEG,cv2.VideoWriter_fourcc('M','J','P','G'), fps, sizes)

    for frame in frames:
        out.write(frame)


def make_image(img):
    q_im = img.data.mul(255.).clamp(0,255).round()
    im = q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

def files_to_videoTensor(path , downscale=1.):
    from PIL import Image
    files = sorted(os.listdir(path))
    print(len(files))
    images = [torch.Tensor(np.asarray(Image.open(os.path.join(input_video , f)))).type(torch.uint8) for f in files]
    print(images[0].shape)
    videoTensor = torch.stack(images)
    return videoTensor

def video_to_tensor(video):

    videoTensor , _ , md = read_video(video)
    fps = md["video_fps"]
    print(fps)
    return videoTensor

def video_transform(videoTensor , downscale=1):
    
    T , H , W = videoTensor.size(0), videoTensor.size(1) , videoTensor.size(2)
    downscale = int(downscale * 8)
    resizes = 8*(H//downscale) , 8*(W//downscale)
    transforms = torchvision.transforms.Compose([ToTensorVideo() , Resize(resizes)])
    videoTensor = transforms(videoTensor)
    
    # resizes = 720,1280
    print("Resizing to %dx%d"%(resizes[0] , resizes[1]) )
    return videoTensor , resizes

if args.is_folder:
    videoTensor = files_to_videoTensor(input_video , args.downscale)
else:
    videoTensor = video_to_tensor(input_video)

idxs = torch.Tensor(range(len(videoTensor))).type(torch.long).view(1,-1).unfold(1,size=nbr_frame,step=1).squeeze(0)
videoTensor , resizes = video_transform(videoTensor , args.downscale)
print("Video tensor shape is , " , videoTensor.shape)

frames = torch.unbind(videoTensor , 1)
n_inputs = len(frames)
width = n_outputs + 1

outputs = [] ## store the input and interpolated frames

outputs.append(frames[idxs[0][1]])

model = model.eval()

for i in tqdm.tqdm(range(len(idxs))):
    idxSet = idxs[i]
    inputs = [frames[idx_].cuda().unsqueeze(0) for idx_ in idxSet]
    with torch.no_grad():
        outputFrame = model(inputs)   
    outputFrame = [of.squeeze(0).cpu().data for of in outputFrame]
    outputs.extend(outputFrame)
    outputs.append(inputs[2].squeeze(0).cpu().data)

new_video = [make_image(im_) for im_ in outputs]

write_video_cv2(new_video , output_video , args.output_fps , (resizes[1] , resizes[0]))

print("Writing to " , output_video.split(".")[0] + ".mp4")
os.system('ffmpeg -hide_banner -loglevel warning -i %s %s'%(output_video , output_video.split(".")[0] + ".mp4"))
os.remove(output_video)
