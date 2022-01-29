
## Semantic Labels
python ../modules/data_prep.py --input $DATA_ROOT/synthetic/train/labcmp --output $DATA_ROOT/synthetic/train/labcmp_resize --option resize
python ../modules/data_prep.py --input $DATA_ROOT/synthetic/train/labcmp_resize --output $DATA_ROOT/synthetic/train/labcmp_concrete_resize --option group
python ../modules/data_prep.py --input $DATA_ROOT/files_train.csv --output $DATA_ROOT/splits/ --option split
python ../modules/data_prep.py --input $DATA_ROOT/synthetic/train/labcmp_resize --output $DATA_ROOT/synthetic/train/labcmp_binary --option binary

## Damage Labels
python ../modules/data_prep.py --input $DATA_ROOT/synthetic/train/labdmg --output $DATA_ROOT/synthetic/train/labdmg_resize --option resize
python ../modules/data_prep.py --input $DATA_ROOT/synthetic/train/labcmp_resize --output $DATA_ROOT/synthetic/train/labcmp_concrete_resize --option group
python ../modules/data_prep.py --input $DATA_ROOT/files_train.csv --output $DATA_ROOT/splits/ --option split
python ../modules/data_prep.py --input $DATA_ROOT/synthetic/train/labcmp_resize --output $DATA_ROOT/synthetic/train/labcmp_binary --option binary
