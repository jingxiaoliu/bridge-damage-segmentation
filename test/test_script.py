#!/bin/bash
#
#SBATCH --job-name=icshm_cmp_vit
#SBATCH --output=icshm_cmp_vit.out
#SBATCH --error=icshm_cmp_vit.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liujx@stanford.edu
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH -G 2
#SBATCH --gpus-per-node 2

date

module load py-pytorch/1.6.0_py36
pip3 install mmcv_full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
pip3 install git+https://github.com/open-mmlab/mmsegmentation.git
pip3 install tqdm
pip3 install scikit-image

cd ~/research/icshm/ic-shm2021-p1/test/

python3 test_cmp_check.py --nw pspnet --nss 0 --nse 0 --task single

python3 test_cmp.py --nw pspnet --nss 1 --nse 2 --task single
python3 test_cmp.py --nw hrnet --nss 1 --nse 2 --task single
python3 test_cmp.py --nw ocrnet --nss 1 --nse 2 --task single
python3 test_cmp.py --nw resnest --nss 1 --nse 2 --task single
python3 test_cmp.py --nw swin --nss 1 --nse 1 --task single
python3 test_cmp.py --nw pspnet --nss 1 --nse 2 --task single --split 1
python3 test_cmp.py --nw hrnet --nss 1 --nse 2 --task single --split 1
python3 test_cmp.py --nw ocrnet --nss 1 --nse 2 --task single --split 1
python3 test_cmp.py --nw resnest --nss 1 --nse 2 --task single --split 1
python3 test_cmp.py --nw swin --nss 1 --nse 1 --task single --split 1
python3 test_cmp.py --nw pspnet --nss 1 --nse 2 --task single --split 2
python3 test_cmp.py --nw hrnet --nss 1 --nse 2 --task single --split 2
python3 test_cmp.py --nw ocrnet --nss 1 --nse 2 --task single --split 2
python3 test_cmp.py --nw resnest --nss 1 --nse 2 --task single --split 2
python3 test_cmp.py --nw swin --nss 1 --nse 1 --task single --split 2
python3 test_cmp.py --nw pspnet --nss 1 --nse 2 --task single --split 3
python3 test_cmp.py --nw hrnet --nss 1 --nse 2 --task single --split 3
python3 test_cmp.py --nw ocrnet --nss 1 --nse 2 --task single --split 3
python3 test_cmp.py --nw resnest --nss 1 --nse 2 --task single --split 3
python3 test_cmp.py --nw swin --nss 1 --nse 1 --task single --split 3
python3 test_cmp.py --nw pspnet --nss 1 --nse 2 --task single --split 4
python3 test_cmp.py --nw hrnet --nss 1 --nse 2 --task single --split 4
python3 test_cmp.py --nw ocrnet --nss 1 --nse 2 --task single --split 4
python3 test_cmp.py --nw resnest --nss 1 --nse 2 --task single --split 4
python3 test_cmp.py --nw swin --nss 1 --nse 1 --task single --split 4

python3 test_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 test_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 1
python3 test_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 1
python3 test_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 1
python3 test_cmp.py --nw swin --nss 0 --nse 0 --task single --split 1
python3 test_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 2
python3 test_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 2
python3 test_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 2
python3 test_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 2
python3 test_cmp.py --nw swin --nss 0 --nse 0 --task single --split 2
python3 test_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 3
python3 test_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 3
python3 test_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 3
python3 test_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 3
python3 test_cmp.py --nw swin --nss 0 --nse 0 --task single --split 3
python3 test_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 4
python3 test_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 4
python3 test_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 4
python3 test_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 4
python3 test_cmp.py --nw swin --nss 0 --nse 0 --task single --split 4
python3 test_cmp.py --nw hrnet --nss 0 --nse 0 --task mode
python3 test_cmp.py --nw hrnet --nss 0 --nse 0 --task label

python3 test_cmp_dmg.py --nw pspnet --nss 0 --nse 1 --task single
python3 test_cmp_dmg.py --nw hrnet --nss 0 --nse 1 --task single
python3 test_cmp_dmg.py --nw ocrnet --nss 0 --nse 1 --task single
python3 test_cmp_dmg.py --nw resnest --nss 0 --nse 1 --task single
python3 test_cmp_dmg.py --nw swin --nss 0 --nse 0 --task single
python3 test_cmp_dmg.py --nw hrnet --nss 0 --nse 1 --task mode

python3 test_puretex.py --nw pspnet --nss 0 --nse 0 --task single
python3 test_puretex.py --nw hrnet --nss 0 --nse 0 --task single
python3 test_puretex.py --nw ocrnet --nss 0 --nse 0 --task single
python3 test_puretex.py --nw resnest --nss 0 --nse 0 --task single
python3 test_puretex.py --nw swin --nss 0 --nse 0 --task single
python3 test_puretex.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 test_puretex.py --nw hrnet --nss 0 --nse 0 --task single --split 1
python3 test_puretex.py --nw ocrnet --nss 0 --nse 0 --task single --split 1
python3 test_puretex.py --nw resnest --nss 0 --nse 0 --task single --split 1
python3 test_puretex.py --nw swin --nss 0 --nse 0 --task single --split 1
python3 test_puretex.py --nw hrnet --nss 0 --nse 0 --task mode
python3 test_puretex.py --nw hrnet --nss 0 --nse 0 --task label

python3 test_dmg.py --nw pspnet --nss 0 --nse 0 --task single
python3 test_dmg.py --nw hrnet --nss 0 --nse 0 --task single
python3 test_dmg.py --nw ocrnet --nss 0 --nse 0 --task single
python3 test_dmg.py --nw resnest --nss 0 --nse 0 --task single
python3 test_dmg.py --nw swin --nss 0 --nse 0 --task single
python3 test_dmg.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 test_dmg.py --nw hrnet --nss 0 --nse 0 --task single --split 1
python3 test_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --split 1
python3 test_dmg.py --nw resnest --nss 0 --nse 0 --task single --split 1
python3 test_dmg.py --nw swin --nss 0 --nse 0 --task single --split 1
python3 test_dmg.py --nw pspnet --nss 0 --nse 0 --task single --split 2
python3 test_dmg.py --nw hrnet --nss 0 --nse 0 --task single --split 2
python3 test_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --split 2
python3 test_dmg.py --nw resnest --nss 0 --nse 0 --task single --split 2
python3 test_dmg.py --nw swin --nss 0 --nse 0 --task single --split 2
python3 test_dmg.py --nw hrnet --nss 0 --nse 0 --task mode
python3 test_dmg.py --nw hrnet --nss 0 --nse 0 --task label

python3 val_cmp_prev.py --nw pspnet --nss 0 --nse 0 --task single
python3 val_cmp_prev.py --nw hrnet --nss 0 --nse 0 --task single
python3 val_cmp_prev.py --nw ocrnet --nss 0 --nse 0 --task single
python3 val_cmp_prev.py --nw resnest --nss 0 --nse 0 --task single
python3 val_cmp_prev.py --nw swin --nss 0 --nse 0 --task single
python3 val_cmp_prev.py --nw hrnet --nss 0 --nse 0 --task mode
python3 val_cmp_prev.py --nw hrnet --nss 0 --nse 0 --task label

python3 val_cmp_slice.py --nw pspnet --nss 0 --nse 0 --task single
python3 val_cmp_slice.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 val_cmp_slice.py --nw pspnet --nss 0 --nse 0 --task single --split 2
python3 val_cmp_slice.py --nw pspnet --nss 0 --nse 0 --task single --split 3
python3 val_cmp_slice.py --nw pspnet --nss 0 --nse 0 --task single --split 4


python3 val_cmp_512.py --nw pspnet --nss 0 --nse 0 --task single
python3 val_cmp_512.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 val_cmp_512.py --nw pspnet --nss 0 --nse 0 --task single --split 2
python3 val_cmp_512.py --nw pspnet --nss 0 --nse 0 --task single --split 3
python3 val_cmp_512.py --nw pspnet --nss 0 --nse 0 --task single --split 4
python3 val_cmp_512.py --nw hrnet --nss 0 --nse 0 --task label

python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single
python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 2
python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 3
python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 4
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 1
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 1
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 1
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --split 1
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 2
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 2
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 2
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --split 2
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 3
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 3
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 3
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --split 3
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 4
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 4
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 4
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --split 4
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task mode
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task label

python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --multiscale
python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_cmp.py --nw pspnet --nss 0 --nse 0 --task single --split 4 --multiscale
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --multiscale
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --multiscale
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --multiscale
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --multiscale
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task single --split 4 --multiscale
python3 val_cmp.py --nw ocrnet --nss 0 --nse 0 --task single --split 4 --multiscale
python3 val_cmp.py --nw resnest --nss 0 --nse 0 --task single --split 4 --multiscale
python3 val_cmp.py --nw swin --nss 0 --nse 0 --task single --split 4 --multiscale
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task mode
python3 val_cmp.py --nw hrnet --nss 0 --nse 0 --task label

python3 val_cmp_visualization.py --nw pspnet --nss 0 --nse 0 --task single

python3 val_dmg_512.py --nw pspnet --nss 0 --nse 0 --task single
python3 val_dmg_512.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 val_dmg_512.py --nw pspnet --nss 0 --nse 0 --task single --split 2
python3 val_dmg_512.py --nw pspnet --nss 0 --nse 0 --task single --split 3
python3 val_dmg_512.py --nw hrnet --nss 0 --nse 0 --task label


python3 val_dmg_puretex.py --nw pspnet --nss 0 --nse 0 --task single --multiscale
python3 val_dmg_puretex.py --nw pspnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_dmg_puretex.py --nw pspnet --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_dmg_puretex.py --nw pspnet --nss 0 --nse 0 --task single --split 3 --multiscale



python3 val_dmg.py --nw pspnet --nss 0 --nse 0 --task single
python3 val_dmg.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 val_dmg.py --nw pspnet --nss 0 --nse 0 --task single --split 2
python3 val_dmg.py --nw pspnet --nss 0 --nse 0 --task single --split 3
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task single
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task single --split 1
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task single --split 2
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task single --split 3
python3 val_dmg.py --nw ocrnet --nss 0 --nse 0 --task single
python3 val_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --split 1
python3 val_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --split 2
python3 val_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --split 3
python3 val_dmg.py --nw resnest --nss 0 --nse 0 --task single
python3 val_dmg.py --nw resnest --nss 0 --nse 0 --task single --split 1
python3 val_dmg.py --nw resnest --nss 0 --nse 0 --task single --split 2
python3 val_dmg.py --nw resnest --nss 0 --nse 0 --task single --split 3
python3 val_dmg.py --nw swin --nss 0 --nse 0 --task single
python3 val_dmg.py --nw swin --nss 0 --nse 0 --task single --split 1
python3 val_dmg.py --nw swin --nss 0 --nse 0 --task single --split 2
python3 val_dmg.py --nw swin --nss 0 --nse 0 --task single --split 3
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task mode
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task label

python3 val_dmg.py --nw pspnet --nss 0 --nse 0 --task single --multiscale
python3 val_dmg.py --nw pspnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_dmg.py --nw pspnet --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_dmg.py --nw pspnet --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task single --multiscale
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --multiscale
python3 val_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_dmg.py --nw ocrnet --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_dmg.py --nw resnest --nss 0 --nse 0 --task single --multiscale
python3 val_dmg.py --nw resnest --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_dmg.py --nw resnest --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_dmg.py --nw resnest --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_dmg.py --nw swin --nss 0 --nse 0 --task single --multiscale
python3 val_dmg.py --nw swin --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_dmg.py --nw swin --nss 0 --nse 0 --task single --split 2 --multiscale
python3 val_dmg.py --nw swin --nss 0 --nse 0 --task single --split 3 --multiscale
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task mode
python3 val_dmg.py --nw hrnet --nss 0 --nse 0 --task label

python3 val_dmg_visualization.py --nw pspnet --nss 0 --nse 0 --task single


python3 val_puretex_512.py --nw hrnet --nss 0 --nse 0 --task single
python3 val_puretex_512.py --nw hrnet --nss 0 --nse 0 --task single --split 1

python3 val_puretex.py --nw pspnet --nss 0 --nse 0 --task single
python3 val_puretex.py --nw pspnet --nss 0 --nse 0 --task single --split 1
python3 val_puretex.py --nw hrnet --nss 0 --nse 0 --task single
python3 val_puretex.py --nw hrnet --nss 0 --nse 0 --task single --split 1
python3 val_puretex.py --nw ocrnet --nss 0 --nse 0 --task single
python3 val_puretex.py --nw ocrnet --nss 0 --nse 0 --task single --split 1
python3 val_puretex.py --nw resnest --nss 0 --nse 0 --task single
python3 val_puretex.py --nw swin --nss 0 --nse 0 --task single
python3 val_puretex.py --nw resnest --nss 0 --nse 0 --task single --split 1
python3 val_puretex.py --nw swin --nss 0 --nse 0 --task single --split 1
python3 val_puretex.py --nw hrnet --nss 0 --nse 0 --task mode
python3 val_puretex.py --nw hrnet --nss 0 --nse 0 --task label

python3 val_puretex.py --nw pspnet --nss 0 --nse 0 --task single --multiscale
python3 val_puretex.py --nw pspnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_puretex.py --nw hrnet --nss 0 --nse 0 --task single --multiscale
python3 val_puretex.py --nw hrnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_puretex.py --nw ocrnet --nss 0 --nse 0 --task single --multiscale
python3 val_puretex.py --nw ocrnet --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_puretex.py --nw resnest --nss 0 --nse 0 --task single --multiscale
python3 val_puretex.py --nw swin --nss 0 --nse 0 --task single --multiscale
python3 val_puretex.py --nw resnest --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_puretex.py --nw swin --nss 0 --nse 0 --task single --split 1 --multiscale
python3 val_puretex.py --nw hrnet --nss 0 --nse 0 --task mode
python3 val_puretex.py --nw hrnet --nss 0 --nse 0 --task label


python3 val_puretex_visualization.py --nw ocrnet --nss 0 --nse 0 --task single


DATA_ROOT=/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset
python3 ../modules/data_prep.py --input $DATA_ROOT/synthetic_puretex/labdmg --output $DATA_ROOT/synthetic_puretex/labdmg_resize --option --option resize
python3 ../modules/data_prep.py --input $DATA_ROOT/imp_aug_gt_10/img --output $DATA_ROOT/splits/augcmp_resampling --option splitbycase_augcmp --resampling True
python3 ../modules/data_prep.py --input $DATA_ROOT/dmg_aug_gt_100/img --output $DATA_ROOT/splits/augdmg_resampling --option splitbycase_augdmg --resampling False
