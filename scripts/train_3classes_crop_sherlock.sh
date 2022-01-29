#!/bin/bash
#SBATCH --job-name=ocr_dmg
#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --gpus-per-node 2
#SBATCH -G 2
#SBATCH -C GPU_MEM:16GB
#SBATCH --output=%x_%J.out
#SBATCH --error=%x_%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liujx@stanford.edu
#echo commands to stdout
set -x
date

source /home/users/liujx/.bashrc
conda activate open-mmlab 
# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

cd ~/research/icshm/ic-shm2021-p1/damage_detection
# run a pre-compiled program which is already in your project space

python -m torch.distributed.launch --nproc_per_node=2 train_3classes_crop_sherlock.py --nw pspnet --cp ../configs/pspnet/pspnet_r101-d8_512x512_4x4_160k_coco-stuff164k.py --bs 16 --iter 200000 --log_iter 1000 --eval_iter 2000 --checkpoint_iter 5000 --distributed --ohem --multi_loss --job_name $SLURM_JOB_NAME --split_id 0
