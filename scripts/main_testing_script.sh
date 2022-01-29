# First move checkpoints and configuration files into dedicated folders.
# For example, we put trained components segmentaion checkpoints and configuration files into:
#	$DATA_ROOT/
#			valid_cmp_cp/
#					hrnet/		# networks: hrnet, pspnet, ocrnet, resnest, swin
#						0/		# cross-validation number: 0-9


cd bridge-damage-segmentation/
DATA_ROOT="/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset"

# Testing phase
# Testing component segmentation model
# hrnet
python3 ./test/test.py --nw hrnet --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_cmp/hrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# pspnet
python3 ./test/test.py --nw hrnet --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_cmp/pspnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# ocrnet
python3 ./test/test.py --nw hrnet --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_cmp/ocrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# resnest
python3 ./test/test.py --nw hrnet --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_cmp/resnest/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# swin
python3 ./test/test.py --nw hrnet --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_cmp/swin/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# ensemble learning
# Change subsets starting and ending numbers for cross validation
python3 ./test/test.py --nss 0 --nse 0 --task mode --save_path $DATA_ROOT/cmp_test --type cmp --split_csv $DATA_ROOT/splits/test_cmp.txt 



# Masking out non-column components for testing data
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/test --output $DATA_ROOT/img_syn_raw/test_mask --split_csv $DATA_ROOT/splits/test_dmg.txt --lbl_dir $DATA_ROOT/cmp_test_dmg/ --option mask_imgs

# 
