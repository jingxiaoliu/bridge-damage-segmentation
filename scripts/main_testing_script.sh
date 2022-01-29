

cd bridge-damage-segmentation/
DATA_ROOT="/home/icshm_data/data_proj1/Tokaido_dataset" # Change this to your own data root

# Testing phase
# Testing component segmentation model
# First move checkpoints and configuration files into dedicated folders.
# For example, we put trained components segmentaion checkpoints and configuration files into:
#	$DATA_ROOT/
#			valid_checkpoints_cmp/
#								hrnet/			# networks: hrnet, pspnet, ocrnet, resnest, swin
#									0/		# cross-validation number: 0-9

# hrnet
python3 ./test/test.py --nw hrnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/hrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# pspnet
python3 ./test/test.py --nw pspnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/pspnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# ocrnet
python3 ./test/test.py --nw ocrnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/ocrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# resnest
python3 ./test/test.py --nw resnest --task single --cp $DATA_ROOT/valid_checkpoints_cmp/resnest/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# swin
python3 ./test/test.py --nw swin --task single --cp $DATA_ROOT/valid_checkpoints_cmp/swin/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_cmp.txt --save_path $DATA_ROOT/cmp_test --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_cmp.txt --type cmp --width 640 --height 360
# ensemble learning
# Change subsets starting and ending numbers for cross validation
python3 ./test/test.py --nss 0 --nse 0 --task mode --save_path $DATA_ROOT/cmp_test --type cmp --split_csv $DATA_ROOT/splits/test_cmp.txt

# Testing damage segmentation in real scene
# First move checkpoints and configuration files into dedicated folders.
# For example, we put trained components segmentaion checkpoints and configuration files into:
#	$DATA_ROOT/
#			valid_checkpoints_dmg/
#								hrnet/			# networks: hrnet, pspnet, ocrnet, resnest, swin
#									0/		# cross-validation number: 0-9

# Predicts structural components labels for training and validation data using hrnet
# Change cross-validation subsets numbers if having cross-validation
python3 ./test/test.py --nw hrnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/hrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_dmg.txt --save_path $DATA_ROOT/cmp_test_dmg --img_dir img_syn_raw/test_resize --ann_dir synthetic/train_labcmp --split splits/test_dmg.txt --type cmp --width 640 --height 360
# Masking out non-column components for testing data
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/test --output $DATA_ROOT/img_syn_raw/test_mask --split_csv $DATA_ROOT/splits/test_dmg.txt --lbl_dir $DATA_ROOT/cmp_test_dmg/ --option mask_imgs

# hrnet
python3 ./test/test.py --nw hrnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/hrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_dmg.txt --save_path $DATA_ROOT/dmg_test --img_dir img_syn_raw/test --ann_dir synthetic/labdmg --split splits/test_dmg.txt --type dmg --width 1920 --height 1080
# pspnet
python3 ./test/test.py --nw pspnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/pspnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_dmg.txt --save_path $DATA_ROOT/dmg_test --img_dir img_syn_raw/test --ann_dir synthetic/labdmg --split splits/test_dmg.txt --type dmg --width 1920 --height 1080
# ocrnet
python3 ./test/test.py --nw ocrnet --task single --cp $DATA_ROOT/valid_checkpoints_cmp/ocrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_dmg.txt --save_path $DATA_ROOT/dmg_test --img_dir img_syn_raw/test --ann_dir synthetic/labdmg --split splits/test_dmg.txt --type dmg --width 1920 --height 1080
# resnest
python3 ./test/test.py --nw resnest --task single --cp $DATA_ROOT/valid_checkpoints_cmp/resnest/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_dmg.txt --save_path $DATA_ROOT/dmg_test --img_dir img_syn_raw/test --ann_dir synthetic/labdmg --split splits/test_dmg.txt --type dmg --width 1920 --height 1080
# swin
python3 ./test/test.py --nw swin --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_cmp/swin/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_dmg.txt --save_path $DATA_ROOT/dmg_test --img_dir img_syn_raw/test --ann_dir synthetic/labdmg --split splits/test_dmg.txt --type dmg --width 1920 --height 1080
# ensemble learning
# Change subsets starting and ending numbers for cross validation
python3 ./test/test.py --nss 0 --nse 0 --task mode --save_path $DATA_ROOT/dmg_test/ --type dmg --split_csv $DATA_ROOT/splits/test_dmg.txt

# Testing damage segmentation in pure texture
# First move checkpoints and configuration files into dedicated folders.
# For example, we put trained components segmentaion checkpoints and configuration files into:
#	$DATA_ROOT/
#			valid_checkpoints_puretex/
#								hrnet/			# networks: hrnet, pspnet, ocrnet, resnest, swin
#									0/		# cross-validation number: 0-9

# Predicts structural components labels for training and validation data using hrnet
# hrnet
python3 ./test/test.py --nw hrnet --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_puretex/hrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_puretex.txt --save_path $DATA_ROOT/puretex_test --img_dir images_puretex --ann_dir synthetic_puretex/labdmg --split splits/test_puretex.txt --type puretex --width 1920 --height 1080
# pspnet
python3 ./test/test.py --nw pspnet --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_puretex/pspnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_puretex.txt --save_path $DATA_ROOT/puretex_test --img_dir images_puretex --ann_dir synthetic_puretex/labdmg --split splits/test_puretex.txt --type puretex --width 1920 --height 1080
# ocrnet
python3 ./test/test.py --nw ocrnet --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_puretex/ocrnet/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_puretex.txt --save_path $DATA_ROOT/puretex_test --img_dir images_puretex --ann_dir synthetic_puretex/labdmg --split splits/test_puretex.txt --type puretex --width 1920 --height 1080
# resnest
python3 ./test/test.py --nw resnest --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_puretex/resnest/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_puretex.txt --save_path $DATA_ROOT/puretex_test --img_dir iimages_puretex --ann_dir synthetic_puretex/labdmg --split splits/test_puretex.txt --type puretex --width 1920 --height 1080
# swin
python3 ./test/test.py --nw swin --task single --cp /home/groups/noh/icshm_data/valid_checkpoints_puretex/swin/0/ --dr $DATA_ROOT --split_csv $DATA_ROOT/splits/test_puretex.txt --save_path $DATA_ROOT/puretex_test --img_dir images_puretex --ann_dir synthetic_puretex/labdmg --split splits/test_puretex.txt --type puretex --width 1920 --height 1080
# ensemble learning
# Change subsets starting and ending numbers for cross validation
python3 ./test/test.py --nss 0 --nse 0 --task mode --save_path $DATA_ROOT/puretex_test --type puretex --split_csv $DATA_ROOT/splits/test_puretex.txt
