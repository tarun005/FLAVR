#!/bin/bash

#SBATCH --job-name=unet_kin_nbrframe_10_nbrwidth_1

#SBATCH --output=/checkpoint/tarun05/slurm_logs/kineticsFinal/%x.out

#SBATCH --error=/checkpoint/tarun05/slurm_logs/kineticsFinal/%x.err

#SBATCH --partition=priority

#SBATCH --nodes=1

#SBATCH --gres=gpu:8

#SBATCH --cpus-per-task=80

#SBATCH --time=72:00:00

#SBATCH --mail-user=tarun05@fb.com

#SBATCH --mail-type=begin,end,fail,requeue # mail once the job finishes

#SBATCH --signal=USR1@300

#SBATCH --open-mode=append

#SBATCH --comment="Internship end"

module purge
source ~/init.sh
source activate TSR

export nf=10
export nw=1

export model=unet_18
cd ~/projects/SuperResolution_Video/
export exp_name=unet_kin_nbrframe_${nf}_nbrwidth_${nw}
export save_loc=/checkpoint/tarun05/saved_models_final/kinetics/${exp_name}/
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

srun --label /private/home/tarun05/.conda/envs/TSR/bin/python main.py --exp_name ${exp_name} --batch_size 64 --test_batch_size 64 --dataset kinetics --model ${model} --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root /datasets01/kinetics/070618/  --checkpoint_dir /checkpoint/tarun05/ --frames_per_clip 60 --nbr_frame ${nf} --nbr_width 1 --joinType concat --num_workers 16 --upmode transpose --n_outputs 1
