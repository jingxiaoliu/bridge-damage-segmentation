#!/bin/bash
#SBATCH --job-name=ocr
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

python -m torch.distributed.launch --nproc_per_node=2 train_3classes.py --nw ocrnet --cp ../configs/ocrnet/ocrnet_hr18_512x512_80k_ade20k.py --bs 8 --data_root /home/groups/noh/icshm_data/data_proj1/Tokaido_dataset/ --work_dir /home/groups/noh/icshm_data/checkpoints/ --iter 200000 --log_iter 10 --eval_iter 10 --checkpoint_iter 10 --distributed --job_name $SLURM_JOB_NAME
