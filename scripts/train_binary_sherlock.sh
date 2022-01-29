#!/bin/bash
#SBATCH --job-name=binary_%a
#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --gpus-per-node 2
#SBATCH -G 4
#SBATCH --output=binary_%a_%J.out
#SBATCH --error=binary_%a_%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yujiew@andrew.cmu.edu

#echo commands to stdout
set -x
date

module load py-pytorch/1.6.0_py36
pip3 install mmcv_full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
pip3 install git+https://github.com/open-mmlab/mmsegmentation.git
pip3 install tqdm

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

cd ~/research/icshm/ic-shm2021-p1/segmentation
# run a pre-compiled program which is already in your project space

python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 train_binary.py --nw ocrnet --cp ~/research/icshm/ic-shm2021-p1/configs/ocrnet/ocrnet_hr18_512x512_80k_ade20k.py --bs 4 --data_root /home/groups/noh/icshm_data/data_proj1/Tokaido_dataset --work_dir /home/groups/noh/icshm_data/data_proj1/checkpoints --category $SLURM_ARRAY_TASK_ID
