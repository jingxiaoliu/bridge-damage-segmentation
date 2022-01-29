cd bridge-damage-segmentation/
DATA_ROOT="/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset"


# prepare data
# resize data and label
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/train --output $DATA_ROOT/img_syn_raw/train_resize --option resize --width 640 --height 360
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/test --output $DATA_ROOT/img_syn_raw/test_resize --option resize --width 640 --height 360
python3 ./modules/data_prep.py --input $DATA_ROOT/images_puretex --output $DATA_ROOT/images_puretex_resize --option resize --width 640 --height 360
python3 ./modules/data_prep.py --input $DATA_ROOT/synthetic/train/labcmp --output $DATA_ROOT/synthetic/train/labcmp_resize --option resize --width 1920 --height 1080
python3 ./modules/data_prep.py --input $DATA_ROOT/synthetic/train/labdmg --output $DATA_ROOT/synthetic/train/labdmg_resize --option resize --width 1920 --height 1080
python3 ./modules/data_prep.py --input $DATA_ROOT/synthetic_puretex/labdmg --output $DATA_ROOT/synthetic_puretex/labdmg_resize --option resize --width 1920 --height 1080
# split data into train and test
python3 ./modules/data_prep.py --input $DATA_ROOT/files_train.csv --output $DATA_ROOT/splits --option splitbycase --data_root $DATA_ROOT --resampling 1
python3 ./modules/data_prep.py --input $DATA_ROOT/files_puretex_train.csv --output $DATA_ROOT/splits --option split_puretex --data_root $DATA_ROOT --resampling 1

# Training phase
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM /apis/train_components.py --nw hrnet --cp $DATA_ROOT/checkpoints ~/research/icshm/ic-shm2021-p1/configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py --bs 2 --distributed --multi_loss --ohem --job_name hrnet_ohem$1 --split_id $1

# Testing phase
# Masking out non-column components
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/train --output $DATA_ROOT/img_syn_raw/code_test/train_mask --split_csv $DATA_ROOT/splits/train_dmg0.txt --lbl_dir $DATA_ROOT/cmp_train_dmg/ --option mask_imgs
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/train --output $DATA_ROOT/img_syn_raw/code_test/train_mask --split_csv $DATA_ROOT/splits/val_dmg0.txt --lbl_dir $DATA_ROOT/cmp_val_dmg/ --option mask_imgs
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/test --output $DATA_ROOT/img_syn_raw/code_test/test_mask --split_csv $DATA_ROOT/splits/train_dmg0.txt --lbl_dir /home/groups/noh/icshm_data/cmp_train_dmg/ --option mask_imgs
$DATA_ROOT/cmp_train_dmg/ 