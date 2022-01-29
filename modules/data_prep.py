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
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--nearest", type=bool, default=True)
parser.add_argument("--resampling", type=bool, default=False)
parser.add_argument("--test", action='store_true')
args = parser.parse_args()


def mask_imgs(out_dir,out_dir_test):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir_test):
        os.makedirs(out_dir_test)

    ori_dir = "/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset/img_syn_raw/train/"
    ori_dir_test = "/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset/img_syn_raw/test/"

    lbl_dir = "/home/groups/noh/icshm_data/cmp_val_dmg/"
    lbl_dir_test = "/home/groups/noh/icshm_data/cmp_test_dmg/ensemble/"
    ori_lbl_dir = "/home/groups/noh/icshm_data/cmp_train_dmg/"

    ori_dir_img_list = os.listdir(ori_dir)
    ori_dir_img_list.sort()
    ori_dir_test_img_list = os.listdir(ori_dir_test)
    ori_dir_test_img_list.sort()
    lbl_dir_img_list = os.listdir(lbl_dir)
    lbl_dir_img_list.sort()
    lbl_dir_test_img_list = os.listdir(lbl_dir_test)
    lbl_dir_test_img_list.sort()
    ori_lbl_dir_img_list = os.listdir(ori_lbl_dir)
    ori_lbl_dir_img_list.sort()

    split_csv = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset/splits/cmp_resampling/val_dmg0.txt'
    val_images = []

    # Ignore depth for the moment, we could preprocess depth in the future
    with open(split_csv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            val_images.append(line.strip('\n'))
    val_images = list(val_images)
    print(val_images[0])
    print("Testing images for damage detection: ", len(val_images))

    split_csv = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset/splits/cmp_resampling/train_dmg0.txt'
    train_images = []

    # Ignore depth for the moment, we could preprocess depth in the future
    with open(split_csv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            train_images.append(line.strip('\n'))
    train_images = list(train_images)
    print(train_images[0])
    print("Testing images for damage detection: ", len(train_images))

    split_csv = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset/splits/test_dmg.txt'
    test_images = []

    # Ignore depth for the moment, we could preprocess depth in the future
    with open(split_csv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            test_images.append(line.strip('\n'))
    test_images = list(test_images)
    print(test_images[0])
    print("Testing images for damage detection: ", len(test_images))

    # for i in tqdm(range(len(val_images)), desc="Processing ..."):
    #     img = np.array(cv2.imread(os.path.join(ori_dir, val_images[i]+'_Scene.png'), cv2.IMREAD_UNCHANGED))
    #     lbl = np.tile(np.expand_dims(np.array(Image.open(os.path.join(lbl_dir, val_images[i]+'.bmp')).resize((1920,1080))),2),(1,1,3)).astype(np.uint8)
    #     if img is None:
    #         print('Wrong path:', os.path.join(ori_dir, val_images[i]))
    #     else:
    #         img[lbl != 4] = 0
    #         cv2.imwrite(os.path.join(out_dir, val_images[i]+'_Scene.png'), img)

    for i in tqdm(range(len(train_images)), desc="Processing ..."):
        img = np.array(cv2.imread(os.path.join(ori_dir, train_images[i]+'_Scene.png'), cv2.IMREAD_UNCHANGED))
        lbl = np.tile(np.expand_dims(np.array(Image.open(os.path.join(ori_lbl_dir, train_images[i]+'.bmp')).resize((1920,1080))),2),(1,1,3)).astype(np.uint8)
        if img is None:
            print('Wrong path:', os.path.join(ori_dir, train_images[i]))
        else:
            img[lbl != 4] = 0
            cv2.imwrite(os.path.join(out_dir, train_images[i]+'_Scene.png'), img)

    # for i in tqdm(range(len(test_images)), desc="Processing ..."):
    #     img = np.array(cv2.imread(os.path.join(ori_dir_test, test_images[i]+'_Scene.png'), cv2.IMREAD_UNCHANGED))
    #     lbl = np.tile(np.expand_dims(np.array(Image.open(os.path.join(lbl_dir_test, test_images[i]+'.bmp')).resize((1920,1080))),2),(1,1,3)).astype(np.uint8)
    #     if img is None:
    #         print('Wrong path:', os.path.join(ori_dir_test, test_images[i]))
    #     else:
    #         img[lbl != 4] = 0
    #         cv2.imwrite(os.path.join(out_dir_test, test_images[i]+'_Scene.png'), img)


def slice_imgs(in_dir, out_dir, nearest=True):
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
            out_img = img[0:360,0:640]
            cv2.imwrite(os.path.join(out_dir, '1' ,img_name), out_img)
            out_img = img[0:360,640:1280]
            cv2.imwrite(os.path.join(out_dir, '2' ,img_name), out_img)
            out_img = img[0:360,1280:1960]
            cv2.imwrite(os.path.join(out_dir, '3' ,img_name), out_img)
            out_img = img[360:720,0:640]
            cv2.imwrite(os.path.join(out_dir, '4' ,img_name), out_img)
            out_img = img[360:720,640:1280]
            cv2.imwrite(os.path.join(out_dir, '5' ,img_name), out_img)
            out_img = img[360:720,1280:1960]
            cv2.imwrite(os.path.join(out_dir, '6' ,img_name), out_img)
            out_img = img[720:1080,0:640]
            cv2.imwrite(os.path.join(out_dir, '7' ,img_name), out_img)
            out_img = img[720:1080,640:1280]
            cv2.imwrite(os.path.join(out_dir, '8' ,img_name), out_img)
            out_img = img[720:1080,1280:1960]
            cv2.imwrite(os.path.join(out_dir, '9' ,img_name), out_img)

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

def verify_img(in_dir, out_dir, num_images=20):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        print("Input directory {0} not exists".format(out_dir))
        return
    img_list = os.listdir(in_dir)
    img_list.sort()
    img_list = img_list[:num_images]
    for img_name in img_list:
        img1 = cv2.imread(os.path.join(in_dir, img_name), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(os.path.join(out_dir, img_name), cv2.IMREAD_UNCHANGED)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), cv2.INTER_NEAREST)
        res = np.subtract(img1, img2)
        res = res.flatten()
        count = np.sum(res>0)
        nc1 = len(np.unique(img1.flatten()))
        nc2 = len(np.unique(img2.flatten()))
        print("NumClasses: {0}|{1}  Image {2} diff pixels: {3}".format(nc1, nc2, img_name, count))

def group_labels(in_dir, out_dir):
    # Grouping components for damage detection.
    # [0, 1, 5, 6, 7, 8] are [Undefined, Nonbridge, Nonstructural, Rail, Sleeper, Others]
    # [2, 3, 4] are [Slab, Beam, Column]
    # The output would be: [0, 1] = [Nonconcrete, concrete]
    new_definition = [[0, 1, 5, 6, 7, 8], [2, 3, 4]]
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_list = os.listdir(in_dir)
    img_list.sort()
    for img_name in tqdm(img_list, desc="Processing ..."):
        img = cv2.imread(os.path.join(in_dir, img_name), cv2.IMREAD_UNCHANGED)
        concrete_mask = (img >= 2) & (img <= 4)
        nonconcrete_mask = ~concrete_mask
        img[concrete_mask] = 1
        img[nonconcrete_mask] = 0
        cv2.imwrite(os.path.join(out_dir, img_name), img)

def one_vs_rest(in_dir, out_dir):
    # Grouping components for damage detection.
    # [0, 1, 5, 6, 7, 8] are [Undefined, Nonbridge, Nonstructural, Rail, Sleeper, Others]
    # [2, 3, 4] are [Slab, Beam, Column]
    # The output would be: [0, 1] = [Nonconcrete, concrete]
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    sub_folders = [os.path.join(out_dir, "{0}".format(x+1)) for x in range(8)]
    if not os.path.exists(out_dir):
        for sub_folder in sub_folders:
            os.makedirs(sub_folder)

    img_list = os.listdir(in_dir)
    img_list.sort()
    for img_name in tqdm(img_list, desc="Processing ..."):
        img = cv2.imread(os.path.join(in_dir, img_name), cv2.IMREAD_UNCHANGED)
        ori = img.copy()
        for i in range(8):
            selected_mask = (ori == (i+1))
            rest_mask = ~selected_mask
            img[selected_mask] = 1
            img[rest_mask] = 0
            cv2.imwrite(os.path.join(sub_folders[i], img_name), img)

def count_pixels(in_dir, out_dir):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    pixel_counts = np.zeros(9, dtype='float64')
    img_list = os.listdir(in_dir)
    img_list.sort()
    
    for img_name in tqdm(img_list, desc="Processing ..."):
    # for img_name in img_list:
        img = cv2.imread(os.path.join(in_dir, img_name), cv2.IMREAD_UNCHANGED)
        vals, cnts = np.unique(img, return_counts=True)
        pixel_counts[vals] = pixel_counts[vals] + cnts

    print(pixel_counts)
    np.savetxt(os.path.join(out_dir, "pixel_counts.txt"), pixel_counts)

def split(in_dir, out_dir, test=False, train_ratio=0.9, seed=13):
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
    
    if not test:
        random.seed(seed)
        random.shuffle(train_images_cmp)
        random.shuffle(train_images_dmg)

        with open(os.path.join(out_dir, 'train_cmp.txt'), 'w') as f:
            # select the ratio portion as training set
            train_length = int(len(train_images_cmp) * train_ratio)
            f.writelines(line + '\n' for line in train_images_cmp[:train_length])
        with open(os.path.join(out_dir, 'val_cmp.txt'), 'w') as f:
            # select the rest as validation set
            f.writelines(line + '\n' for line in train_images_cmp[train_length:])
        with open(os.path.join(out_dir, 'train_dmg.txt'), 'w') as f:
            # select the ratio portion as training set
            train_length = int(len(train_images_dmg) * train_ratio)
            f.writelines(line + '\n' for line in train_images_dmg[:train_length])
        with open(os.path.join(out_dir, 'val_dmg.txt'), 'w') as f:
            # select the rest as validation set
            f.writelines(line + '\n' for line in train_images_dmg[train_length:])
    else:
        with open(os.path.join(out_dir, 'test_cmp.txt'), 'w') as f:
            f.writelines(line+'\n' for line in train_images_cmp)
        
        with open(os.path.join(out_dir, 'test_dmg.txt'), 'w') as f:
            f.writelines(line+'\n' for line in train_images_dmg)

def splitbycase_augdmg(in_dir, out_dir, seed=13, resampling=False):
    data_root = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset'
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        print(1)
        os.makedirs(out_dir)
    train_images_dmg = []
    for file in glob.glob(in_dir+'/*.png'):
        train_images_dmg.append(file.split("/")[-1].strip("_Scene.png"))
    train_images_dmg = list(train_images_dmg)
    
    random.seed(seed)

    cases = list(np.arange(0,175,1))
    random.shuffle(cases)
    linspace = list(np.arange(0,175,10))
    for i in range(9):
        train_images_dmg_train = []
        train_images_dmg_val = []
        case_id = cases[linspace[i]:linspace[i+1]]
        for name in train_images_dmg:
            if name.split('_')[1][4:] in str(case_id):
                train_images_dmg_val.append(name)
            else:
                train_images_dmg_train.append(name)
        with open(os.path.join(out_dir, 'train_dmg'+str(i)+'.txt'), 'w') as f:
            # select the ratio portion as training set
            n=1
            for line in train_images_dmg_train:
                repeat = 1
                if resampling:
                    file_name = data_root+'/dmg_aug_gt_100/seg/'+line+'.bmp'
                    img =  np.array(Image.open(file_name))
                    er_labels = np.where(img==3)[0]
                    # print(er_labels)
                    if len(er_labels) >= 10:
                        n+=1
                        repeat = 2
                for r in range(repeat):
                    f.writelines(line + '\n')
            # print(n)
        with open(os.path.join(out_dir, 'val_dmg'+str(i)+'.txt'), 'w') as f:
            # select the rest as validation set
            f.writelines(line + '\n' for line in train_images_dmg_val)


def splitbycase_augcmp(in_dir, out_dir, seed=13, resampling=False):
    data_root = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset'
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_images_cmp = []
    for file in glob.glob(in_dir+'/*.png'):
        train_images_cmp.append(file.split("/")[-1].strip("_Scene.png"))
    train_images_cmp = list(train_images_cmp)
    
    random.seed(seed)

    cases = list(np.arange(0,175,1))
    random.shuffle(cases)
    linspace = list(np.arange(0,175,10))
    for i in range(9):
        train_images_cmp_train = []
        train_images_cmp_val = []
        case_id = cases[linspace[i]:linspace[i+1]]
        for name in train_images_cmp:
            if name.split('_')[1][4:] in str(case_id):
                train_images_cmp_val.append(name)
            else:
                train_images_cmp_train.append(name)
        with open(os.path.join(out_dir, 'train_cmp'+str(i)+'.txt'), 'w') as f:
            # select the ratio portion as training set
            n=1
            for line in train_images_cmp_train:
                repeat = 1
                if resampling:
                    file_name = data_root+'/imp_aug_gt_10/seg/'+line+'.bmp'
                    img =  np.array(Image.open(file_name))
                    er_labels = np.where(img==7)[0]
                    # print(er_labels)
                    if len(er_labels) >= 10:
                        n+=1
                        repeat = 2
                for r in range(repeat):
                    f.writelines(line + '\n')
            # print(n)
        with open(os.path.join(out_dir, 'val_cmp'+str(i)+'.txt'), 'w') as f:
            # select the rest as validation set
            f.writelines(line + '\n' for line in train_images_cmp_val)

def splitbycase(in_dir, out_dir, seed=13, resampling=False):
    data_root = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset'
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
    for i in range(9):
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
                    er_labels = np.where(img==7)[0]
                    # print(er_labels)
                    if len(er_labels) >= 10:
                        n+=1
                        repeat = 10
                    er_labels1 = np.where(img==5)[0]
                    if len(er_labels1) >= 10:
                        n+=1
                        repeat = 10
                for r in range(repeat):
                    f.writelines(line + '\n')
            # print(n)
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
                    er_labels = np.where(img==3)[0]
                    # print(er_labels)
                    if len(er_labels) >= 10:
                        n+=1
                        repeat = 3
                for r in range(repeat):
                    f.writelines(line + '\n')
            print(n)
        with open(os.path.join(out_dir, 'val_dmg'+str(i)+'.txt'), 'w') as f:
            # select the rest as validation set
            f.writelines(line + '\n' for line in train_images_dmg_val)
        
def split_puretex(in_dir, out_dir, test=False, train_ratio=0.9, seed=13, resampling=False):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_images = []

    data_root = '/home/groups/noh/icshm_data/data_proj1/Tokaido_dataset'
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
        
def class_stat(in_dir, out_dir, num_classes=8):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_list = os.listdir(in_dir)
    img_list.sort()

def color_stat(in_dir, out_dir):
    if not os.path.exists(in_dir):
        print("Input directory {0} not exists".format(in_dir))
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_list = os.listdir(in_dir)
    img_list.sort()

    # Number of pixels per channel
    n = 0
    # Sum of colors per channel, channel order: BGR
    s1 = [0.0] * 3
    # Sum of color*2 per channel, channel order: BGR
    s2 = [0.0] * 3

    for img_name in tqdm(img_list, desc="Processing ..."):
        img = cv2.imread(os.path.join(in_dir, img_name), cv2.IMREAD_UNCHANGED)
        n = n + len(img[:, :, 0])
        # Update
        for i in range(3):
            s1[i] = s1[i] + np.sum(img[:, :, i].flatten())
            s2[i] = s2[i] + np.sum(np.square(img[:, :, i].flatten()))
    mean = [x * 1.0/n for x in s1]
    std = [np.sqrt((n * y - x * x)/(n * (n-1))) for x,y in zip(s1, s2)]
    print("Mean BGR: ", mean)
    print("Std BGR: std", std)

    with open(os.path.join(out_dir, "bgr_stat.txt"), 'w') as f:
        f.write(" ".join(mean))
        f.write("\n")
        f.write(" ".join(std))

def resample_by_class(root_dir, out_dir, width, height):
    if not os.path.exists(root_dir):
        print("Incorrect root directory.")
        return
   
    oimg_dir = os.path.join(out_dir, "img")
    oseg_dir = os.path.join(out_dir, "seg")
    if not os.path.exists(oimg_dir):
        os.makedirs(oimg_dir, exist_ok=True)
        os.makedirs(oseg_dir, exist_ok=True)
    # Modify this for augmenting different dataset.
    img_dir = os.path.join(root_dir, "img_syn_raw", "train")
    img_list = os.listdir(img_dir)
    seg_dir = os.path.join(root_dir, "synthetic", "train", "labdmg")
    img_list.sort()
    
    for img_name in tqdm(img_list, desc="Processing ..."):
        seg_name = img_name.replace("_Scene.png", ".bmp")
        if not os.path.exists(os.path.join(seg_dir, seg_name)):
            continue
        seg = cv2.imread(os.path.join(seg_dir, seg_name), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_UNCHANGED)

        # The first image to save
        img_small = cv2.resize(img, (width, height), cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(oimg_dir, img_name), img_small)
        cv2.imwrite(os.path.join(oseg_dir, seg_name), seg)
        # Select class of interest
        selected = (seg == 2) | (seg == 3)
        if((not np.any(selected)) or (np.sum(selected) < 100)):
            continue
        # raw_seg = cv2.resize(seg, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)        
        width_list = np.linspace(1920, width, 9, dtype=np.int32)
        height_list = np.linspace(1080, height, 9, dtype=np.int32)
        for idx, (iw, ih) in enumerate(zip(width_list, height_list)):
            img_tmp = cv2.resize(img, (iw, ih), cv2.INTER_NEAREST)
            seg_tmp = cv2.resize(seg, (iw, ih), cv2.INTER_NEAREST)
            max_score = -1
            max_x = 0
            max_y = 0

            # img.shape = (height, width, channel) 
            for y in range(0, img_tmp.shape[0], 20):
                for x in range(0, img_tmp.shape[1], 20):
                    seg_patch = seg_tmp[y:y+height, x:x+width]
                    if (seg_patch.shape[0] < height) or (seg_patch.shape[1] < width):
                        continue
                    score = np.sum((seg_patch == 2) | (seg_patch == 3))
                    if(score > max_score):
                        max_x = x
                        max_y = y
                        max_score = score
            img_patch = img_tmp[max_y:max_y+height, max_x:max_x+width]
            seg_patch = seg_tmp[max_y:max_y+height, max_x:max_x+width]

            img_patch_name = img_name.replace("_Scene.png", "_"+str(idx)+"_Scene.png")
            seg_patch_name = seg_name.replace(".bmp", "_"+str(idx)+".bmp")
            
            cv2.imwrite(os.path.join(oimg_dir, img_patch_name), img_patch)
            cv2.imwrite(os.path.join(oseg_dir, seg_patch_name), seg_patch)

def main():
    print(args.option)
    if(args.option == "resize"):
        resize_imgs(args.input, args.output, args.width, args.height)
    if(args.option == "verify"):
        verify_img(args.input, args.output)
    if(args.option == "group"):
        group_labels(args.input, args.output)
    if(args.option == "count"):
        count_pixels(args.input, args.output)
    if(args.option == "split"):
        split(args.input, args.output, args.test)
    if(args.option == "color"):
        color_stat(args.input, args.output)
    if(args.option == "binary"):
        one_vs_rest(args.input, args.output)
    if(args.option == "split_puretex"):
        split_puretex(args.input, args.output, args.test, resampling=args.resampling)
    if(args.option == "splitbycase"):
        splitbycase(args.input, args.output,resampling=args.resampling)
    if(args.option == "resample"):
        resample_by_class(args.input, args.output, args.width, args.height)
    if(args.option == "splitbycase_augcmp"):
        splitbycase_augcmp(args.input, args.output,resampling=args.resampling)
    if(args.option == "splitbycase_augdmg"):
        splitbycase_augdmg(args.input, args.output,resampling=args.resampling)
    if(args.option == "slice_imgs"):
        slice_imgs(args.input, args.output)
    if(args.option == "mask_imgs"):
        mask_imgs(args.input, args.output)
if __name__ == "__main__":
    main()
