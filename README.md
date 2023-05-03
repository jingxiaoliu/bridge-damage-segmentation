# Bridge-damage-segmentation

This is the code repository for the paper [A hierarchical semantic segmentation framework for computer-vision-based bridge column damage detection](https://www.researchgate.net/publication/370467357_A_hierarchical_semantic_segmentation_framework_for_computer_vision-based_bridge_damage_detection) submitted to the [IC-SHM Challenge 2021](https://sail.cive.uh.edu/ic-shm2021/). The semantic segmentation framework used in this paper leverages importance sampling, semantic mask, and multi-scale test time augmentation to achieve a 0.836 IoU for scene component segmentation and a 0.467 IoU for concrete damage segmentation on the [Tokaido Dataset](). The framework was implemented on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) using Python.

# Highlights

# Models used in the framework
### Backbones
- HRNet
- Swin
- ResNest

### Decoder Heads
- PSPNet
- UperNet
- OCRNet

# Performance
The following table reports IoUs for structural component segmentation.

|            Architecture                          | Slab  | Beam  | Column | Non-structural | Rail  | Sleeper | Average |
|--------------------------------------------------|-------|-------|--------|----------------|-------|---------|---------|
| Ensemble                                         | 0.891 | 0.880 |  0.859 |      0.660     | 0.623 |  0.701  |  0.785  |
| Ensemble + Importance sampling                   | 0.915 | 0.912 |  0.958 |      0.669     | 0.618 |  0.892  |  0.827  |
| Ensemble + Importance sampling + Multi-scale TTA | 0.924 | 0.929 |  0.965 |      0.681     | 0.621 |  0.894  |  0.836  |

The following table reports IoUs for damage segmentation of pure texture images.
| Architecture                                     | Concrete damage | Exposed rebar | Average |
|--------------------------------------------------|-----------------|---------------|---------|
| Ensemble                                         |      0.356      |     0.536     |  0.446  |
| Ensemble + Importance sampling                   |      0.708      |     0.714     |  0.711  |
| Ensemble + Importance sampling + Multi-scale TTA |      0.698      |     0.727     |  0.712  |

The following table reports IoUs for damage segmentation of real scene images.
| Architecture                                     | Concrete damage | Exposed rebar | Average |
|--------------------------------------------------|-----------------|---------------|---------|
| Ensemble                                         |      0.235      |     0.365     |  0.300  |
| Ensemble + Importance sampling                   |      0.340      |     0.557     |  0.448  |
| Ensemble + Importance sampling + Multi-scale TTA |      0.350      |     0.583     |  0.467  |
| Ensemble + Importance sampling + Multi-scale TTA + Mask |      0.379      |     0.587     |  0.483  |

# Environment
The code is developed under the following configurations.
- Hardware: >= 2 GPUs for training, >= 1 GPU for testing. The script supports sbatch training and testing on computer clusters.
- Software: 
  - System: Ubuntu 16.04.3 LTS 
  - CUDA >= 10.1
- Dependencies:
  - [Conda](https://www.anaconda.com/): This is optional, but we suggest using conda to configure the environment.
  - [Pytorch >= 1.6.0](https://pytorch.org/)
  - [MMCV](https://github.com/open-mmlab/mmcv)
  - [MMSeg](https://github.com/open-mmlab/mmsegmentation)
  - [OpenCV >= 4.5.0](https://github.com/opencv/opencv/releases)
  - tqdm

# Usage
### Environment
1. Install conda and create a conda environment

    ```sh
    $ conda create -n open-mmlab
    $ source activate open-mmlab
    $ conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
    ```

2. Install mmcv-full

    ```sh
    $ pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
    ```

3. Install mmsegmentation

    ```sh
    $ pip install git+https://github.com/open-mmlab/mmsegmentation.git
    ```

4. Install other dependencies
    ```sh
    $ pip install opencv, tqdm, numpy, scipy
    ```
    
5. Download the Tokaido dataset from [IC-SHM Challenge 2021](https://sail.cive.uh.edu/ic-shm2021/).

### Training
1. Example single model training using multiple GPUs
    ```sh
    $ python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --master_port=$RANDOM ./apis/train_damage_real.py \
      --nw hrnet \
      --cp $CHECKPOINT_DIR \
      --dr $DATA_ROOT \
      --conf $MODEL_CONFIG \
      --bs 16 \
      --train_split $TRAIN_SPLIT_PATH \
      --val_split $VAL_SPLIT_PATH \
      --width 1920 \
      --height 1080 \
      --distributed \
      --iter 100000 \
      --log_iter 10000 \
      --eval_iter 10000 \
      --checkpoint_iter 10000 \
      --multi_loss \
      --ohem \
      --job_name dmg
    ```
2. Example shell script for preparing the whole dataset and train all models for the whole pipeline.
    ```sh
    $ ./scripts/main_training_script.sh
    ```

### Evlauation
1. Eval one model
    ```sh
    $ python3 ./test/test.py \
      --nw hrnet \
      --task single \
      --cp $CONFIG_PATH \
      --dr $DATA_ROOT \
      --split_csv $RAW_CSV_PATH \
      --save_path $OUTPOUT_DIR \
      --img_dir $INPUT_IMG_DIR \
      --ann_dir $INPUT_GT_DIR \
      --split $TEST_SPLIT_PATH \
      --type cmp \
      --width 640 \
      --height 360
    ```
2. Example shell script for testing the whole pipeline and generate the output using the IC-SHM Challenge format.
    ```sh
    $ ./scripts/main_testing_script.sh
    ```

3. Visualization (Add the `--cmp` flag when visualizing components.)
    ```sh
    $ ./modules/viz_label.py \
      --input $SEG_DIR
      --output $OUTPUT_DIR
      --raw_input $IMG_DIR
      --cmp 
    ```

# Reference
If you find the code useful, please cite the following paper.
```
Liu, J., Wei, Y., Chen, B. & Noh, H. (2023). A hierarchical semantic segmentation framework for computer vision-based bridge damage detection. SMART STRUCTURES AND SYSTEMS 31(4):325-334
```
