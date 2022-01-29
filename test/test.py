import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataset,build_dataloader
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmseg.apis import multi_gpu_test, single_gpu_test
from PIL import Image
import torchvision
import mmcv
import mmseg
print(mmseg.__version__)
# Parse document
from os import path, rename, listdir
import glob
from scipy import stats
CUDA_LAUNCH_BLOCKING=1
torch.manual_seed(0)
import argparse

# Testing arguments, check the help message for details.
parser = argparse.ArgumentParser()
parser.add_argument("--nw", type=str, default='pspnet', help="Network name.")
parser.add_argument("--nss", type=int, default=0, help="Cross validation subsets starting number.")
parser.add_argument("--nse", type=int, default=1, help="Cross validation subsets ending number.")
parser.add_argument("--task", type=str, default='single', help="Task name")
parser.add_argument("--cp", type=str, required=False, help="Checkpoint path.")
parser.add_argument("--dr", type=str, required=False, help="Data root.")
parser.add_argument("--split_csv", type=str, required=True, help="Split file.")
parser.add_argument("--save_path", type=str, required=True, help="Prediction save path.")
parser.add_argument("--img_dir", type=str, required=False, help="Image file directory.")
parser.add_argument("--ann_dir", type=str, required=False, help="Label directory.")
parser.add_argument("--split", type=str, required=False, help="Split file in root.")
parser.add_argument("--slide", action='store_true', help="Testing with sliding window")
parser.add_argument("--type", type=str, required=True, help="Testing type.")
parser.add_argument("--width", type=int, default=640, help='Image width. ')
parser.add_argument("--height", type=int, default=360, help='Image height. ')
args = parser.parse_args()

if __name__ == '__main__':
	checkpoint_path = args.cp
	test_save_path = args.save_path
	network = args.nw
	task_name = args.task
	# Origanize dataset
	data_root = args.dr

	split_csv = args.split_csv
	test_images = []
	with open(split_csv, 'r') as f:
		lines = f.readlines()
		for line in lines:
			test_images.append(line.strip('\n'))
	test_images = list(test_images)
	print(test_images[0])
	print("Testing images: ", len(test_images))

	# Eval using a single model
	if task_name == 'single':
		if args.type == 'dmg':
			classes = ('Undefined','Undamaged', 'ConcreteDamage', 'ExposedRebar')
			palette = [[0,0,0], [128, 128, 128], [129, 127, 38], [120, 69, 125]]
		elif args.type == 'cmp':			
			classes = ('Undefined', 'Nonbridge', 'Slab', 'Beam', 'Column', 'Nonstructural', 'Rail', 'Sleeper', 'Other')
			palette = [[0,0,0], [128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
					   [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]
		@DATASETS.register_module()
		class TokaidoDataset(CustomDataset):
			CLASSES = classes
			PALETTE = palette
			def __init__(self, split, **kwargs):
				super().__init__(img_suffix='_Scene.png', seg_map_suffix='.bmp', 
								 split=split, **kwargs)
				assert path.exists(self.img_dir) and self.split is not None
		
		cfg_file = glob.glob(checkpoint_path+'*.py')[0]
		cfg = Config.fromfile(cfg_file)
		if args.slide:
			cfg.model.test_cfg.mode='slide'
			cfg.model.test_cfg.stride = (128,128)
			cfg.model.test_cfg.crop_size = (256,256)
		cfg.model.pretrained = None
		cfg.data.test.test_mode = True
		cfg.data.test.data_root = data_root
		cfg.data.test.img_dir = args.img_dir
		cfg.data.test.ann_dir= args.ann_dir
		cfg.data.test.split = args.split

		# Multi-scale TTA.
		cfg.data.test.pipeline[1]=dict(
			type='MultiScaleFlipAug',
			img_scale=(args.width, args.height),
			img_ratios = [0.5,1.0,1.5,2.0],
			flip=False,
			transforms=[
				dict(type='Resize', keep_ratio=True),
				dict(type='RandomFlip'),
				dict(
					type='Normalize',
					mean=[123.675, 116.28, 103.53],
					std=[58.395, 57.12, 57.375],
					to_rgb=True),
				dict(type='ImageToTensor', keys=['img']),
				dict(type='Collect', keys=['img'])
			])
		cfg.test_pipeline[1]=dict(
			type='MultiScaleFlipAug',
			img_scale=(args.width, args.height),
			img_ratios = [0.5,1.0,1.5,2.0],
			flip=False,
			transforms=[
				dict(type='Resize', keep_ratio=True),
				dict(type='RandomFlip'),
				dict(
					type='Normalize',
					mean=[123.675, 116.28, 103.53],
					std=[58.395, 57.12, 57.375],
					to_rgb=True),
				dict(type='ImageToTensor', keys=['img']),
				dict(type='Collect', keys=['img'])
			])

		dataset = build_dataset(cfg.data.test)
		data_loader = build_dataloader(
			dataset,
			samples_per_gpu=1,
			workers_per_gpu=cfg.data.workers_per_gpu,
			dist=False,
			shuffle=False)

		# build the model and load checkpoint
		model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
		cp_file = glob.glob(checkpoint_path+'/*.pth')[0]
		checkpoint = load_checkpoint(model, cp_file, map_location='cpu')
		model.CLASSES = classes
		model.PALETTE = palette
		model.cfg = cfg     
		model.to('cuda')
		model.eval()
		model = MMDataParallel(model, device_ids=[0])
		outputs = single_gpu_test(model, data_loader)    
		for j in range(len(outputs)):
			save_path = test_save_path+network+'/'+str(k)+'/'		
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			save_file = save_path+test_images[j]+'.bmp'
			img = Image.fromarray(outputs[j].astype(np.uint8)).resize((640,360))
			img.save(save_file)

	# Eval using majority vote
	elif task_name == 'mode':
		networks = ['hrnet','ocrnet','pspnet','resnest','swin']
		# networks = ['hrnet','ocrnet','pspnet']
		num_subsets = args.nse - args.nss + 1
		for i in tqdm(range(len(test_images))):
			tmp_tensor = np.empty((len(networks)*num_subsets,360,640))
			idx = 0
			for j in range(len(networks)):
				for k in range(args.nss,args.nse+1):
					tmp_tensor[idx,:,:] = Image.open(test_save_path+networks[j]+'/'+str(k)+'/'+test_images[i]+'.bmp')
					idx+=1
			result = np.reshape(stats.mode(np.reshape(tmp_tensor[0:len(networks)*num_subsets,:,:],(len(networks)*num_subsets,360*640)),axis=0).mode,(360,640))
			result = Image.fromarray(result.astype(np.uint8))

			result.save(test_save_path+'ensemble/'+test_images[i]+'.bmp')
