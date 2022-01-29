import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import random
from PIL import Image
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--option", type=str, required=True,
                    help="Use resize|count|group|verify|split|resample to preprocess datasets.")
parser.add_argument("--input", type=str, required=True,
                    help="Input directory, usually the directory of images.")
parser.add_argument("--output", type=str, required=True,
                    help="Output directory if there is any output")
parser.add_argument("--data_root", type=str, required=False,
                    help="Taikoda data root")
parser.add_argument("--split_csv", type=str, required=False,
                    help="Split file")
parser.add_argument("--lbl_dir", type=str, required=False,
                    help="Label directory")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--nearest", type=bool, default=True)
parser.add_argument("--resampling", type=bool, default=False)
parser.add_argument("--test", action='store_true')
args = parser.parse_args()

def mask_imgs(in_dir, out_dir, split_csv, lbl_dir):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    split_csv = args.split_csv
    images = []
    with open(split_csv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            images.append(line.strip('\n'))
    images = list(images)
    print(images[0])
    print("Testing images for damage detection: ", len(images))

    for i in tqdm(range(len(images)), desc="Masking validation images..."):
        img = np.array(cv2.imread(os.path.join(in_dir, images[i]+'_Scene.png'), cv2.IMREAD_UNCHANGED))
        lbl = np.tile(np.expand_dims(np.array(Image.open(os.path.join(lbl_dir, images[i]+'.bmp')).resize((1920,1080))),2),(1,1,3)).astype(np.uint8)
        if img is None:
            print('Wrong path:', os.path.join(in_dir, images[i]))
        else:
            img[lbl != 4] = 0
            cv2.imwrite(os.path.join(out_dir, images[i]+'_Scene.png'), img)

def resize_imgs(in_dir, out_dir, width, height, nearest=True):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_list = os.listdir(in_dir)
    img_list.sort()
    for img_name in tqdm(img_list, desc="Processing ..."):
        img = cv2.imread(os.path.join(in_dir, img_name), cv2.IMREAD_UNCHANGED)
        if img is None:
            print('Wrong path:', os.path.join(in_dir, img_name))
        else:
            out_img = cv2.resize(img, (width , height), cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out_dir, img_name), out_img)

def splitbycase(in_dir, out_dir, data_root, seed=13, resampling=False):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_images_cmp = []
    train_images_dmg = []

    with open(in_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.replace("\\", "/").strip("\n").split(",")
            # valid image for cmp training
            if (words[5] == 'True'):
                train_images_cmp.append(os.path.basename(words[0].strip("_Scene.png")))
            if (words[6] == 'True'):
                train_images_dmg.append(os.path.basename(words[0].strip("_Scene.png")))
    train_images_cmp = list(train_images_cmp)
    train_images_dmg = list(train_images_dmg)
    
    random.seed(seed)

    cases = list(np.arange(0,175,1))
    random.shuffle(cases)
    linspace = list(np.arange(0,175,10))
    # 10-fold
    for i in range(10):
        train_images_cmp_train = []
        train_images_cmp_val = []
        train_images_dmg_train = []
        train_images_dmg_val = []
        case_id = cases[linspace[i]:linspace[i+1]]
        for name in train_images_cmp:
            if name.split('_')[1][4:] in str(case_id):
                train_images_cmp_val.append(name)
            else:
                train_images_cmp_train.append(name)
        for name in train_images_dmg:
            if name.split('_')[1][4:] in str(case_id):
                train_images_dmg_val.append(name)
            else:
                train_images_dmg_train.append(name)
        with open(os.path.join(out_dir, 'train_cmp'+str(i)+'.txt'), 'w') as f:
            # select the ratio portion as training set
            n=1
            for line in train_images_cmp_train:
                repeat = 1
                if resampling:
                    file_name = data_root+'/synthetic/train/labcmp/'+line+'.bmp'
                    img =  np.array(Image.open(file_name))
                    # if sleeper
                    er_labels = np.where(img==7)[0]
                    if len(er_labels) >= 10:
                        n+=1
                        repeat = 10
                    # if non-structural
                    er_labels1 = np.where(img==5)[0]
                    if len(er_labels1) >= 10:
                        n+=1
                        repeat = 10
                for r in range(repeat):
                    f.writelines(line + '\n')
        with open(os.path.join(out_dir, 'val_cmp'+str(i)+'.txt'), 'w') as f:
            # select the rest as validation set
            f.writelines(line + '\n' for line in train_images_cmp_val)
        with open(os.path.join(out_dir, 'train_dmg'+str(i)+'.txt'), 'w') as f:
            # select the ratio portion as training set
            for line in train_images_dmg_train:
                repeat = 1
                if resampling:
                    file_name = data_root+'/synthetic/train/labdmg/'+line+'.bmp'
                    img =  np.array(Image.open(file_name))
                    # if exposed rebar
                    er_labels = np.where(img==3)[0]
                    if len(er_labels) >= 10:
                        n+=1
                        repeat = 3
                for r in range(repeat):
                    f.writelines(line + '\n')
        with open(os.path.join(out_dir, 'val_dmg'+str(i)+'.txt'), 'w') as f:
            # select the rest as validation set
            f.writelines(line + '\n' for line in train_images_dmg_val)
        
def split_puretex(in_dir, out_dir, data_root, test=False, train_ratio=0.9, seed=13, resampling=False):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_images = []
    with open(in_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.replace("\\", "/").strip("\n").split(",")
            # valid image 
            train_images.append(os.path.basename(words[0].strip(".png")))
    train_images = list(train_images)
    
    if not test:
        random.seed(seed)
        random.shuffle(train_images)

        with open(os.path.join(out_dir, 'train_puretex.txt'), 'w') as f:
            # select the ratio portion as training set
            train_length = int(len(train_images) * train_ratio)
            for line in train_images[:train_length]:
                repeat = 1
                if resampling:
                    file_name = data_root+'/synthetic_puretex/labdmg/'+line+'.bmp'
                    img =  np.array(Image.open(file_name))
                    er_labels = np.where(img==3)[0]
                    # print(er_labels)
                    if len(er_labels) >= 10:
                        repeat = 5
                for r in range(repeat):
                    f.writelines(line + '\n')
        with open(os.path.join(out_dir, 'val_puretex.txt'), 'w') as f:
            # select the rest as validation set
            f.writelines(line + '\n' for line in train_images[train_length:])
    else:
        with open(os.path.join(out_dir, 'test_puretex.txt'), 'w') as f:
            f.writelines(line+'\n' for line in train_images)
        
def main():
    print(args.option)
    if(args.option == "resize"):
        resize_imgs(args.input, args.output, args.width, args.height)
    if(args.option == "split_puretex"):
        split_puretex(args.input, args.output, args.data_root, args.test, resampling=args.resampling)
    if(args.option == "splitbycase"):
        splitbycase(args.input, args.output, args.data_root, resampling=args.resampling)
    if(args.option == "mask_imgs"):
        mask_imgs(args.input, args.output, args.split_csv, args.lbl_dir)

if __name__ == "__main__":
    main()
