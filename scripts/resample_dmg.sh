#!/bin/bash
#SBATCH --job-name=dmg_aug
#SBATCH -p normal
#SBATCH -t 5:00:00
#SBATCH -c 4
#SBATCH --output=%x_%J.out
#SBATCH --error=%x_%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liujx@stanford.edu
#echo commands to stdout
set -x
date

source /home/users/liujx/.bashrc
conda activate open-mmlab

cd ~/research/icshm/ic-shm2021-p1/modules 

python data_prep.py --input /home/groups/noh/icshm_data/data_proj1/Tokaido_dataset --output /home/groups/noh/icshm_data/data_proj1/Tokaido_dataset/dmg_aug_gt_100 --width 640 --height 360 --option resample
