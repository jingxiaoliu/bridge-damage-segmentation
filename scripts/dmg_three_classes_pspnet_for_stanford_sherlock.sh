#!/bin/bash
#
#SBATCH --job-name=icshm_dmg_pspnet
#SBATCH --output=icshm_dmg_pspnet.out
#SBATCH --error=icshm_dmg_pspnet.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liujx@stanford.edu
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node 2
#SBATCH -c 4
#SBATCH -G 4

date

#module load py-pytorch/1.6.0_py36
#pip3 install mmcv_full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
#pip3 install git+https://github.com/open-mmlab/mmsegmentation.git
#pip3 install tqdm

cd ~/research/icshm/ic-shm2021-p1/damage_detection/

python3 -m torch.distributed.launch --nproc_per_node=2 train_3classes_for_stanford_sherlock.py --nw pspnet --cp ~/research/icshm/ic-shm2021-p1/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py --bs 4
