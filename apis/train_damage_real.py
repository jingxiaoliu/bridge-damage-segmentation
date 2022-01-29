import torch, torchvision
import os
import numpy as np
from PIL import Image
import mmcv
from mmcv import Config
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmcv.runner import get_dist_info, init_dist
import argparse
from datetime import datetime
from random import randint

# Training arguments, check the help message for details.
parser = argparse.ArgumentParser()
parser.add_argument("--nw", type=str, required=True,
                    help="Network name.")
parser.add_argument("--conf", type=str, required=True,help="Config path.")
parser.add_argument("--cp", type=str, required=True,help="checkpoint path.")
parser.add_argument("--bs", type=int, required=True,
                    help="Batch size.")
parser.add_argument("--dr", type=str, required=True, help="Data root.")
parser.add_argument("--local_rank", type=int, help="")
parser.add_argument("--train_split", type=str, required=True, help="Split file for training")
parser.add_argument("--val_split", type=str, required=True, help="Split file for testing")
parser.add_argument("--width", type=int, default=640, help='Image width. ')
parser.add_argument("--height", type=int, default=360, help='Image height. ')
parser.add_argument("--distributed", action='store_true')
parser.add_argument("--resume_from", type=str, help="Resume from a previous checkpoint. Pass the checkpoint path as an argument.")
parser.add_argument("--iter", type=int, default=40000, help='Max number of iterations.')
parser.add_argument("--log_iter", type=int, default=10, help="The interval for logging.")
parser.add_argument("--eval_iter", type=int, default=200, help="Validation interval.")
parser.add_argument("--checkpoint_iter", type=int, default=2000, help="Checkpoint interval.")
parser.add_argument("--learning_rate", type=float, help="Learning rate of the optimizer.")
parser.add_argument("--ohem", action='store_true')
parser.add_argument("--multi_loss", action='store_true')
parser.add_argument("--nlr", type=float, default=1.0, help='Set different learning rate for backbone and head.')
parser.add_argument("--job_name", type=str, default='', help="job name used in sbatch to create folders.")
args = parser.parse_args()

# Concrete segmentation dataset: Two classes only.
classes = ('Undefined','Undamaged', 'ConcreteDamage', 'ExposedRebar')
palette = [[0,0,0], [128, 128, 128], [129, 127, 38], [120, 69, 125]]

@DATASETS.register_module()
class TokaidoDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='_Scene.png', seg_map_suffix='.bmp',
                     split=split, **kwargs)
        assert os.path.exists(self.img_dir) and self.split is not None


# Setup config
data_root = args.dr
num_classes = len(classes)
batch_size = args.bs
image_size = (args.width,args.height)
img_dir = os.path.join('img_syn_raw', 'train_mask')
ann_dir = os.path.join('synthetic', 'train', 'labdmg_resize')
train_split = args.train_split
val_split = args.val_split
checkpoint_dir = args.cp
dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
job_name = args.job_name + "_" + args.nw + "_" + dt_string
network = args.nw

def generate_config(config_path):
    cfg = Config.fromfile(config_path)

    # Since we use ony one GPU, BN is used instead of SyncBN
    class_weight = [0.9939,0.0257,0.9822,0.9981]

    if(args.distributed):
        cfg.norm_cfg = dict(type='SyncBN', requires_grad=True)
    else:
        cfg.norm_cfg = dict(type='BN', requires_grad=True)
    # cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.backbone.pretrained = None
    if network == 'resnest' or network == 'pspnet' or network == 'swin'  or network == 'vit':
        if args.ohem:
            cfg.model.decode_head.sampler = dict(type='OHEMPixelSampler')
        cfg.model.decode_head.norm_cfg = cfg.norm_cfg
        cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
        cfg.model.decode_head.num_classes = num_classes
        # cfg.model.decode_head.loss_decode.class_weight = class_weight
        cfg.model.auxiliary_head.num_classes = num_classes
        # cfg.model.auxiliary_head.loss_decode.class_weight = class_weight
        if args.focal:
            cfg.model.decode_head.loss_decode = dict(
                                                type='FocalLoss',
                                                use_sigmoid=True,
                                                gamma=2.0,
                                                alpha=0.25,
                                                loss_weight=1.0)
            cfg.model.auxiliary_head.loss_decode = dict(
                                                type='FocalLoss',
                                                use_sigmoid=True,
                                                gamma=2.0,
                                                alpha=0.25,
                                                loss_weight=1.0)
        if network == 'swin':
            del cfg.model.backbone.pretrain_style
    elif network == 'ocrnet':
        if args.ohem:
            cfg.model.decode_head[0].sampler = dict(type='OHEMPixelSampler')
            cfg.model.decode_head[1].sampler = dict(type='OHEMPixelSampler') 
        if args.multi_loss:
            cfg.model.decode_head[0].loss_decode = [dict(type='CrossEntropyLoss', loss_name='loss_ce',
                    loss_weight=1.0),
                    dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
            cfg.model.decode_head[1].loss_decode = [dict(type='CrossEntropyLoss', loss_name='loss_ce',
                    loss_weight=1.0),
                    dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]      
        cfg.model.decode_head[0].norm_cfg = cfg.norm_cfg
        cfg.model.decode_head[1].norm_cfg = cfg.norm_cfg
        cfg.model.decode_head[0].num_classes = num_classes
        # cfg.model.decode_head[0].loss_decode.class_weight = class_weight
        cfg.model.decode_head[1].num_classes = num_classes
        # cfg.model.decode_head[1].loss_decode.class_weight = class_weight
    elif network == 'hrnet':
        if args.ohem:
            cfg.model.decode_head.sampler = dict(type='OHEMPixelSampler')
        cfg.model.decode_head.norm_cfg = cfg.norm_cfg     
        cfg.model.decode_head.num_classes = num_classes
        # cfg.model.decode_head.loss_decode.class_weight = class_weight   

    # Modify dataset type and path
    cfg.dataset_type = 'TokaidoDataset'
    cfg.data_root = data_root
    cfg.resume_from = args.resume_from
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = 4

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=image_size, ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=image_size,
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = train_split

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = val_split

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = val_split

    # Set up working dir to save files and logs.
    cfg.work_dir = os.path.join(checkpoint_dir, job_name + "_" + network)
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir, exist_ok=True)

    cfg.runner.max_iters = args.iter
    cfg.log_config.interval = args.log_iter
    cfg.evaluation.interval = args.eval_iter
    cfg.checkpoint_config.interval = args.checkpoint_iter

    # Set seed to facitate reproducing the result
    cfg.seed = randint(0,10000)
    set_random_seed(randint(0,10000), deterministic=False)
    if(args.distributed):
        init_dist('pytorch', **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    else:
        cfg.gpu_ids = [0]

    if args.learning_rate:
        cfg.optimizer = dict(type='SGD', lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)

    # dump config
    cfg.dump(os.path.join(cfg.work_dir,job_name + "_" + network+"_config.py"))

    # Have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')
    return cfg

def main():
    # Build the dataset
    base_config = args.conf
    cfg = generate_config(base_config)
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=True, validate=True,
                    meta=dict())

if __name__ == "__main__":
    main()
