#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-16:2
#SBATCH --output=icshm_cmp_test.out
#SBATCH --error=icshm_cmp_test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yujiew@andrew.cmu.edu

# type 'man sbatch' for more information and options
# this job will ask for 1 full RM node (128 cores) for 5 hours
# this job would potentially charge 640 RM SUs

#echo commands to stdout
set -x

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

cd /ocean/projects/ecs190002p/nba16235/shm
source /jet/home/nba16235/.bashrc 
source activate open-mmlab
# run a pre-compiled program which is already in your project space

python -m torch.distributed.launch --nproc_per_node=2 ic-shm2021-p1/segmentation/train_binary.py --nw ocrnet --cp ic-shm2021-p1/configs/ocrnet/ocrnet_hr18_512x512_80k_ade20k.py --bs 8 --data_root /ocean/projects/ecs190002p/nba16235/shm/Tokaido_dataset --work_dir /ocean/projects/ecs190002p/nba16235/shm/checkpoints --category $1
