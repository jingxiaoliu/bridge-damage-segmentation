#!/bin/bash
#
#SBATCH --job-name=icshm_puretex_swin_%J
#SBATCH --output=icshm_puretex_swin_%J.out
#SBATCH --error=icshm_puretex_swin_%J.err
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

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train_3classes_puretex.py --nw swin --cp ~/research/icshm/ic-shm2021-p1/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py --bs 2 --distributed --multi_loss --ohem --job_name swin_puretex