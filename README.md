# FLAVR

FLAVR is a fast, flow-free frame interpolation method capable of single shot multi-frame prediction. It uses a customized encoder decoder architecture with spatio-temporal convolutions and channel gating to capture and interpolate complex motion trajectories. This repository contains original source code for the paper accepted to CVPR 2021. PDF of the paper is available at [here](https://tarun005.github.io/files/papers/2012.08512.pdf) and more results are available in the [project video.](https://tarun005.github.io/files/papers/2012.08512.pdf)

## Dependencies

We used the following to train and test the model.

- Ubuntu 18.04
- Python==3.7.4
- numpy==1.19.2
- [PyTorch](http://pytorch.org/)==1.5.0, torchvision==0.6.0, cudatoolkit==10.1

## Model

<center><img src="./figures/arch_dia.png" width="90%"></center>

## Training model on Vimeo-90K septuplets

For training your own model on the Vimeo-90K dataset, use the following command. You can download the dataset from [this link](http://toflow.csail.mit.edu/)
``` bash
python main.py --batch_size 32 --test_batch_size 32 --dataset vimeo90K_septuplet --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root <dataset_path> --n_outputs 1
```

Training on GoPro dataset is similar, change `n_outputs` to 7 for 8x interpolation.

## Testing using trained model.

### 2x Interpolation
For testing a pretrained model on Vimeo-90K septuplet validation set, you can download the trained model from [here](https://drive.google.com/drive/folders/1M6ec7t59exOSlx_Wp6K9_njBlLH2IPBC?usp=sharing). Then run the inference using the command
```bash
python test.py --dataset vimeo90K_septuplet --data_root <data_path> --load_from <saved_model> --n_outputs 1
```

### 8x Interpolation
For testing a multiframe interpolation model, download the model trained on GoPro data [here](https://drive.google.com/drive/folders/1Gd2l69j7UC1Zua7StbUNcomAAhmE-xFb?usp=sharing) and use the same command as above, with `n_outputs` changed to 7.

### Time Benchmarking
The testing script, in addition to computing PSNR and SSIM values, will also output the inference time and speed for interpolation. 

### Evaluation on Middleburry

To evaluate on the public benchmark of Middleburry, run the following.
```bash
python Middleburry_Test.py --data_root <data_path> --load_from <model_path> 
```

## SloMo-Filter on custom video

You can use our trained models and apply the slomo filter on your own video (requires OpenCV 4.2.0). Use the following command.
```bash
python interpolate.py --input_video <input_video> --factor <2/4/8> --load_model <model_path>
```

Use a 2x interpolation model if the `factor` is 2 and 8x interpolation model if the `factor` is 8.

## Baseline Models

We also train models for many other previous works on our setting, and provide models for all these methods. Complete benchmarking scripts will also be released soon.

 Method        | PSNR on Vimeo           | Trained Model  |
| ------------- |:-------------:| -----:|
| AdaCoF      | 35.3 | [Model](https://drive.google.com/file/d/19Y2TDZkSbRgNu-OItvqk3qn5cBWGg1RT/view?usp=sharing) |
| QVI      |   35.15    | [Model](https://drive.google.com/file/d/1v2u5diGcvdTLhck8Xwu0baI4zm0JBJhI/view?usp=sharing)   |
| DAIN |   34.19   | [Model](https://drive.google.com/file/d/1RfrwrHoSX_3RIdsoQgPg9IfGAJRhOoEp/view?usp=sharing)  |

## Google Colab

Coming soon ... !

## Acknowledgement

The code is heavily borrowed from Facebook's official [PyTorch video repository](https://github.com/facebookresearch/VMZ) and [CAIN](https://github.com/myungsub/CAIN).

## Cite

If you use our work or the trained models, please consider citing us. 
``` text
@article{kalluri2021flavr,
  title={FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation},
  author={Kalluri, Tarun and Pathak, Deepak and Chandraker, Manmohan and Tran, Du},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
