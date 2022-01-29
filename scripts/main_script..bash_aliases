cd ~/bridge-damge-segmentation/
DATA_ROOT = "/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset"



python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM /apis/train_components.py --nw hrnet --cp DATA_ROOT/checkpoints ~/research/icshm/ic-shm2021-p1/configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py --bs 2 --distributed --multi_loss --ohem --job_name hrnet_ohem$1 --split_id $1
