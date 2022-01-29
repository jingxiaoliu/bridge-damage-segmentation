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

parser = argparse.ArgumentParser()
parser.add_argument("--nw", type=str, default='pspnet',
					help="Network name.")
parser.add_argument("--nss", type=int, default=0,
					help="Subsets starting.")
parser.add_argument("--nse", type=int, default=1,
					help="Subsets ending.")
parser.add_argument("--task", type=str, default='single',
					help="Task name")
parser.add_argument("--split", type=str, default='0',
					help="Split#")
args = parser.parse_args()

def mask2rle(img):
	'''
	img: numpy array, 1 - mask, 0 - background
	Returns run length as string formated
	'''
	pixels = img.T.flatten()
	# pixels= img.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return ' '.join(str(x) for x in runs)

def preprocess(img,labels):
	img = np.array(img)
	lab_1hot = np.zeros((img.shape[0],img.shape[1],len(labels)),dtype=np.bool)
	for i in range(len(labels)):
		lab_1hot[:,:,i] = img==labels[i]
	return lab_1hot

def pred2kaggle(pred_path,csv_read,csv_write,im_col,lab_col,labels,select_col=None):
	# read a csv file associated with the testing data and create Kaggle solution csv file
	# im_col: csv column idx that points to image files
	# lab_col: csv column idx that points to label files
	# select_col: csv column idx that points to selected files (default:None)
	df = pd.read_csv(csv_read,header=None)
	labFiles = list(df[lab_col])
	imFiles = list(df[im_col])
	if select_col is not None:
		imFiles = [imFiles[i] for i in range(len(imFiles)) if df[select_col][i]]
		labFiles = [labFiles[i] for i in range(len(labFiles)) if df[select_col][i]]
	
	data = pd.DataFrame()
	for i in range(len(imFiles)):
		file_name = labFiles[i][2:].split('\\')[-1]
		
		m = preprocess(Image.open(os.path.join(pred_path,file_name)),labels)
		for j in range(len(labels)):
			name = labFiles[i][:-4]+'_'+str(labels[j])
			temp = pd.DataFrame.from_records([
						{
							'ImageId': name,
							'EncodedPixels': mask2rle(m[:,:,j]),  
						}]
					)
			data = pd.concat([data, temp],ignore_index=True)
		print(i)
	if csv_write is not None:
		data.to_csv(csv_write, index=False)
	return data

if __name__ == '__main__':
	checkpoint_path_cmp = '/home/groups/noh/icshm_data/valid_checkpoints_cmp/'
	test_save_path = '/home/groups/noh/icshm_data/cmp_train_dmg/'
	network = args.nw
	task_name = args.task
	# Origanize dataset
	data_root = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset'
	train_file_csv = path.join(data_root, "files_train.csv")
	test_file_csv = path.join(data_root, "files_test.csv")
	test_images_cmp = []
	test_images_dmg = []

	# Ignore depth for the moment, we could preprocess depth in the future
	with open(test_file_csv, 'r') as f:
		lines = f.readlines()
		for line in lines:
			words = line.replace("\\", "/").strip("\n").split(",")
			# valid image for cmp testing
			if(words[5] == 'True'):
				test_images_cmp.append(path.basename(words[0].strip("_Scene.png")))
			if(words[6] == 'True'):
				test_images_dmg.append(path.basename(words[0].strip("_Scene.png")))
	test_images_cmp = list(test_images_cmp)
	test_images_dmg = list(test_images_dmg)
	print(test_images_cmp[0])
	print(test_images_dmg[0])
	print("Testing images for segmentation: ", len(test_images_cmp))
	print("Testing images for damage detection: ", len(test_images_dmg))
	split_csv = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset/splits/cmp_resampling/train_dmg0_'+args.split+'.txt'
	val_images = []

	# Ignore depth for the moment, we could preprocess depth in the future
	with open(split_csv, 'r') as f:
		lines = f.readlines()
		for line in lines:
			val_images.append(line.strip('\n'))
	val_images = list(val_images)
	print(val_images[0])
	print("Testing images for damage detection: ", len(val_images))

	if task_name == 'single':
		# define class and plaette for better visualization
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

		for k in range(args.nss,args.nse+1):
			cfg_file = glob.glob(checkpoint_path_cmp+network+'/'+str(k)+'/*.py')[0]
			cfg = Config.fromfile(cfg_file)
			cfg.model.pretrained = None
			cfg.data.test.test_mode = True
			cfg.data.test.data_root = data_root
			cfg.data.test.img_dir = 'img_syn_raw/train_resize'
			cfg.data.test.ann_dir='synthetic/train/labcmp'
			cfg.data.test.split = 'splits/cmp_resampling/train_dmg0_'+args.split+'.txt'
			dataset = build_dataset(cfg.data.test)
			data_loader = build_dataloader(
				dataset,
				samples_per_gpu=1,
				workers_per_gpu=cfg.data.workers_per_gpu,
				dist=False,
				shuffle=False)

			# build the model and load checkpoint
			model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
			cp_file = glob.glob(checkpoint_path_cmp+network+'/'+str(k)+'/*.pth')[0]
			checkpoint = load_checkpoint(model, cp_file, map_location='cpu')
			model.CLASSES = classes
			model.PALETTE = palette
			model.cfg = cfg     
			model.to('cuda')
			model.eval()
			model = MMDataParallel(model, device_ids=[0])
			outputs = single_gpu_test(model, data_loader)    
			save_path = test_save_path
			if not os.path.exists(save_path):
				os.makedirs(save_path, exist_ok=True)
			for j in range(len(outputs)):
				save_file = test_save_path+'/'+val_images[j]+'.bmp'
				img = Image.fromarray(outputs[j].astype(np.uint8))
				img.save(save_file)

	elif task_name == 'mode':
		networks = ['hrnet','ocrnet','pspnet','resnest','swin']
		# networks = ['hrnet','ocrnet','pspnet']
		num_subsets = args.nse - args.nss + 1
		ensemble_path = test_save_path+'ensemble/'
		if not os.path.exists(ensemble_path):
			os.makedirs(ensemble_path, exist_ok=True)
		for i in tqdm(range(len(test_images_dmg))):
			tmp_tensor = np.empty((len(networks)*num_subsets,360,640))
			idx = 0
			for j in range(len(networks)):
				for k in range(args.nss,args.nse+1):
					tmp_tensor[idx,:,:] = Image.open(test_save_path+networks[j]+'/'+str(k)+'/'+test_images_dmg[i]+'.bmp')
					idx+=1
			result = np.reshape(stats.mode(np.reshape(tmp_tensor[0:len(networks)*num_subsets,:,:],(len(networks)*num_subsets,360*640)),axis=0).mode,(360,640))
			result = Image.fromarray(result.astype(np.uint8))
			result.save(test_save_path+'ensemble/'+test_images_dmg[i]+'.bmp')

