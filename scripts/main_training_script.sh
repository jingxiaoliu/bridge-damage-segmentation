cd bridge-damage-segmentation/
DATA_ROOT="/home/icshm_data/data_proj1/Tokaido_dataset" # Change this to your own data root


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
# if doing cross-validation, change train and validation split file name 
# Train each model for component segmentation
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_components.py --nw hrnet --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py --bs 16 --train_split splits/train_cmp0.txt --val_split splits/val_cmp0.txt --width 640 --height 360 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name cmp
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_components.py --nw pspnet --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/pspnet/pspnet_r101b-d8_512x1024_80k_cityscapes.py --bs 16 --train_split splits/train_cmp0.txt --val_split splits/val_cmp0.txt --width 640 --height 360 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name cmp
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_components.py --nw ocrnet --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py --bs 16 --train_split splits/train_cmp0.txt --val_split splits/val_cmp0.txt --width 640 --height 360 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name cmp
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_components.py --nw resnest --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes.py --bs 4 --train_split splits/train_cmp0.txt --val_split splits/val_cmp0.txt --width 640 --height 360 --distributed --iter 200000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name cmp
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_components.py --nw swin --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py --bs 4 --train_split splits/train_cmp0.txt --val_split splits/val_cmp0.txt --width 640 --height 360 --distributed --iter 200000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name cmp

# Move checkpoints and configuration files into dedicated folders.
# For example, we put trained components segmentaion checkpoints and configuration files into:
#	$DATA_ROOT/
#			valid_checkpoints_cmp/
#								hrnet/			# networks: hrnet, pspnet, ocrnet, resnest, swin
#									0/		# cross-validation number: 0-9

# Predicts structural components labels for training and validation data using hrnet
# Change cross-validation subsets numbers if having cross-validation
python3 ./test/test.py --nw hrnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/hrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/train_dmg0.txt --save_path $DATA_ROOT/cmp_train_dmg --img_dir img_syn_raw/train_resize --ann_dir synthetic/train_labcmp --split splits/train_dmg0.txt --type cmp --width 640 --height 360
python3 ./test/test.py --nw hrnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/hrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/val_dmg0.txt --save_path $DATA_ROOT/cmp_train_dmg --img_dir img_syn_raw/train_resize --ann_dir synthetic/train_labcmp --split splits/val_dmg0.txt --type cmp --width 640 --height 360
# Masking out non-column components for training and validation data
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/train --output $DATA_ROOT/img_syn_raw/train_mask --split_csv $DATA_ROOT/splits/train_dmg0.txt --lbl_dir $DATA_ROOT/cmp_train_dmg/ --option mask_imgs
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/train --output $DATA_ROOT/img_syn_raw/train_mask --split_csv $DATA_ROOT/splits/val_dmg0.txt --lbl_dir $DATA_ROOT/cmp_train_dmg/ --option mask_imgs

# Train each model for damage segmentation in real scene
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_real.py --nw hrnet --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py --bs 16 --train_split splits/train_dmg0.txt --val_split splits/val_dmg0.txt --width 1920 --height 1080 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name dmg
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_real.py --nw pspnet --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/pspnet/pspnet_r101b-d8_512x1024_80k_cityscapes.py --bs 16 --train_split splits/train_dmg0.txt --val_split splits/val_dmg0.txt --width 1920 --height 1080 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name dmg
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_real.py --nw ocrnet --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py --bs 16 --train_split splits/train_dmg0.txt --val_split splits/val_dmg0.txt --width 1920 --height 1080 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name dmg
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_real.py --nw resnest --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes.py --bs 4 --train_split splits/train_dmg0.txt --val_split splits/val_dmg0.txt --width 1920 --height 1080 --distributed --iter 200000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name dmg
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_real.py --nw swin --cp $DATA_ROOT/checkpoints --dr $DATA_ROOT --conf ./configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py --bs 4 --train_split splits/train_dmg0.txt --val_split splits/val_dmg0.txt --width 1920 --height 1080 --distributed --iter 200000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name dmg

# Train each model for damage segmentation in puretex 
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_pure.py --nw hrnet --cp $DATA_ROOT/code_test/checkpoints --dr $DATA_ROOT --conf ./configs/hrnet/fcn_hr48_512x1024_40k_cityscapes.py --bs 16 --train_split splits/train_puretex.txt --val_split splits/val_puretex.txt --width 1920 --height 1080 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name puretex
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_pure.py --nw pspnet --cp $DATA_ROOT/code_test/checkpoints --dr $DATA_ROOT --conf ./configs/pspnet/pspnet_r101b-d8_512x1024_80k_cityscapes.py --bs 16 --train_split splits/train_puretex.txt --val_split splits/val_puretex.txt --width 1920 --height 1080 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name puretex
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_pure.py --nw ocrnet --cp $DATA_ROOT/code_test/checkpoints --dr $DATA_ROOT --conf ./configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py --bs 16 --train_split splits/train_puretex.txt --val_split splits/val_puretex.txt --width 1920 --height 1080 --distributed --iter 100000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name puretex
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_pure.py --nw resnest --cp $DATA_ROOT/code_test/checkpoints --dr $DATA_ROOT --conf ./configs/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes.py --bs 4 --train_split splits/train_puretex.txt --val_split splits/val_puretex.txt --width 1920 --height 1080 --distributed --iter 200000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name puretex
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM ./apis/train_damage_pure.py --nw swin --cp $DATA_ROOT/code_test/checkpoints --dr $DATA_ROOT --conf ./configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py --bs 4 --train_split splits/train_puretex.txt --val_split splits/val_puretex.txt --width 1920 --height 1080 --distributed --iter 200000 --log_iter 10000 --eval_iter 10000 --checkpoint_iter 10000 --multi_loss --ohem --job_name puretex
