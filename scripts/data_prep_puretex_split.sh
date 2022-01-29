DATA_ROOT=/data/shm/Tokaido_dataset
python ../modules/data_prep.py --input $DATA_ROOT/files_puretex_train.csv --output $DATA_ROOT/splits/puretex_resampling --option split_puretex --resampling True
