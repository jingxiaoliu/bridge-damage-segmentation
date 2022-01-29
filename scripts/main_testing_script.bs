cd bridge-damage-segmentation/
DATA_ROOT="/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset"

# Testing phase
# Masking out non-column components
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/train --output $DATA_ROOT/img_syn_raw/train_mask --split_csv $DATA_ROOT/splits/train_dmg0.txt --lbl_dir $DATA_ROOT/cmp_train_dmg/ --option mask_imgs
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/train --output $DATA_ROOT/img_syn_raw/train_mask --split_csv $DATA_ROOT/splits/val_dmg0.txt --lbl_dir $DATA_ROOT/cmp_val_dmg/ --option mask_imgs
python3 ./modules/data_prep.py --input $DATA_ROOT/img_syn_raw/test --output $DATA_ROOT/img_syn_raw/test_mask --split_csv $DATA_ROOT/splits/test_dmg.txt --lbl_dir $DATA_ROOT/cmp_test_dmg/ --option mask_imgs

