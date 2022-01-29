#!/bin/bash
#
#SBATCH --job-name=icshm_dmg_pspnet_%J
#SBATCH --output=icshm_dmg_pspnet_%J.out
#SBATCH --error=icshm_dmg_pspnet_%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liujx@stanford.edu
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH -G 2
#SBATCH --gpus-per-node 2

set -x
date

module load py-pytorch/1.6.0_py36
pip3 install mmcv_full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
pip3 install git+https://github.com/open-mmlab/mmsegmentation.git
pip3 install tqdm

cd ~/research/icshm/ic-shm2021-p1/damage_detection/

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train_3classes_for_stanford_sherlock.py --nw pspnet --cp ~/research/icshm/ic-shm2021-p1/configs/pspnet/pspnet_r101b-d8_512x1024_80k_cityscapes.py --bs 16 --distributed --multi_loss --ohem --job_name pspnet_ohem$1 --split_id $1 --resume_from /home/groups/noh/icshm_data/checkpoints/augdmg/0/pspnet_ohem0_pspnet_20220112-123505_pspnet/latest.pth
