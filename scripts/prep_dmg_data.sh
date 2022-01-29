DATA_ROOT=/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset
python ../damage_detection/crop_image.py --img_dir $DATA_ROOT/img_small/train --seg_dir $DATA_ROOT/synthetic/train/labcmp --dmg_dir $DATA_ROOT/synthetic/train/labdmg --crop_width 256 --crop_height 256 --out_dir $DATA_ROOT/aug_large
python ../damage_detection/crop_image.py --img_dir $DATA_ROOT/img_small/train --seg_dir $DATA_ROOT/synthetic/train/labcmp --dmg_dir $DATA_ROOT/synthetic/train/labdmg --crop_width 128 --crop_height 128 --out_dir $DATA_ROOT/aug_medium
python ../damage_detection/crop_image.py --img_dir $DATA_ROOT/img_small/train --seg_dir $DATA_ROOT/synthetic/train/labcmp --dmg_dir $DATA_ROOT/synthetic/train/labdmg --crop_width 64 --crop_height 64 --out_dir $DATA_ROOT/aug_small
