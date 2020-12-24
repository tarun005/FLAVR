#!/bin/bash

#SBATCH --job-name=unet_R2D18_vimeo_noBN

#SBATCH --output=/checkpoint/tarun05/slurm_logs/vimeoFinal/%x.out

#SBATCH --error=/checkpoint/tarun05/slurm_logs/vimeoFinal/%x.err

#SBATCH --partition=priority

#SBATCH --constraint=volta32gb

#SBATCH --nodes=1

#SBATCH --gres=gpu:8

#SBATCH --cpus-per-task=80

#SBATCH --time=71:00:00

#SBATCH --mail-user=tarun05@fb.com

#SBATCH --mail-type=begin,end,fail,requeue # mail once the job finishes

#SBATCH --signal=USR1@300

#SBATCH --constraint=volta32gb

#SBATCH --open-mode=append

#SBATCH --comment="Internship end"

module purge
source ~/init.sh
source activate TSR

cd ~/projects/SuperResolution_Video/
export n_inputs=4
export n_outputs=1
export exp_name=unet_R2D18_vimeo_noBN
export save_loc=/checkpoint/tarun05/saved_models_final/vimeo90K_septuplet/${exp_name}/
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

srun --label /private/home/tarun05/.conda/envs/TSR/bin/python main.py --exp_name ${exp_name} --batch_size 64 --test_batch_size 64 --dataset vimeo90K_septuplet --model unet_18 --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root /datasets01/Vimeo-90k/070420/vimeo_septuplet/ --checkpoint_dir /checkpoint/tarun05/ --nbr_frame ${n_inputs} --upmode transpose --n_outputs ${n_outputs} --joinType concat

