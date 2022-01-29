#!/bin/bash
#
#SBATCH --job-name=icshm_puretex_hrnet_%J
#SBATCH --output=icshm_puretex_hrnet_%J.out
#SBATCH --error=icshm_puretex_hrnet_%J.err
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

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train_3classes_puretex.py --nw hrnet --cp ~/research/icshm/ic-shm2021-p1/configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py --bs 4 --distributed --multi_loss --ohem --job_name hrnet_puretex