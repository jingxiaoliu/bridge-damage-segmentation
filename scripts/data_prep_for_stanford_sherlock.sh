DATA_ROOT=/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset
python3 ../modules/data_prep.py --input $DATA_ROOT/synthetic/train/labcmp --output $DATA_ROOT/synthetic/train/labcmp_resize --option resize
python3 ../modules/data_prep.py --input $DATA_ROOT/synthetic/train/labdmg --output $DATA_ROOT/synthetic/train/labdmg_resize --option resize
python3 ../modules/data_prep.py --input $DATA_ROOT/files_train.csv --output $DATA_ROOT/splits/ --option split

DATA_ROOT=/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset
python3 ../modules/data_prep.py --input $DATA_ROOT/images_puretex --output $DATA_ROOT/images_puretex_resize --option resize --width 640 --height 360
python3 ../modules/data_prep.py --input $DATA_ROOT/files_puretex_train.csv --output $DATA_ROOT/splits/ --option split_puretex
python3 ../modules/data_prep.py --input $DATA_ROOT/files_puretex_train.csv --output $DATA_ROOT/splits/ --test --option split_puretex

