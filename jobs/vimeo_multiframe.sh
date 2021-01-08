#!/bin/bash

#SBATCH --job-name=unetMF_ip4_op7_resnet18_adobe_gating_noPT

#SBATCH --output=/checkpoint/tarun05/slurm_logs/GoPro/%x.out

#SBATCH --error=/checkpoint/tarun05/slurm_logs/GoPro/%x.err

#SBATCH --constraint=volta32gb

#SBATCH --nodes=1

#SBATCH --gres=gpu:8

#SBATCH --cpus-per-task=80

#SBATCH --time=71:00:00

#SBATCH --mail-user=tarun05@fb.com

#SBATCH --mail-type=begin,end,fail,requeue # mail once the job finishes

#SBATCH --signal=USR1@300

#SBATCH --open-mode=append

module purge
source ~/init.sh
source activate TSR

cd ~/projects/SuperResolution_Video/
export exp_name=unetMF_ip4_op7_resnet18_gopro
export dataset=gopro
export save_loc=/checkpoint/tarun05/saved_models/gopro/${exp_name}/
export data_root=/private/home/tarun05/SuperSloMo/eval_code/data/Adobe240FPS_13frame/

if [ ! -d ${save_loc}/files/ ] 
then
    mkdir -p ${save_loc}/files/
    cp --parents *.py ${save_loc}/files/
    cp --parents model/*.py ${save_loc}/files/
    cp --parents dataset/*.py ${save_loc}/files/
    cp --parents jobs/*.sh ${save_loc}/files/
    cp --parents -r pytorch_msssim ${save_loc}/files/
fi
cd ${save_loc}/files/

srun --label /private/home/tarun05/.conda/envs/TSR/bin/python main.py --exp_name ${exp_name} --batch_size 32 --test_batch_size 32 --dataset ${dataset} --model unet_18 --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root ${data_root} --checkpoint_dir ${save_loc} --upmode transpose --n_outputs 7 --pretrained ${pretrained}
