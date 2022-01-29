import cv2
import numpy as np
import os
from PIL import Image
import argparse

# dmg coding
Concrete_dmg = [128, 0, 0]
Rebar_dmg = [0, 128, 0]
Not_dmg = [0, 0, 128]
Undefined = [0, 0, 0]

# cmp coding
Nonbridge = [0,128,192]
Slab = [128,0,0]
Beam = [192,192,128]
Column = [128,64,128]
Nonstructure = [60,40,222]
Rail = [128,128,0]
Sleeper = [192,128,128]
Other = [64,64,128]

CMP_CMAP = np.array([Undefined, Nonbridge, Slab, Beam, Column, Nonstructure, Rail, Sleeper, Other], dtype=np.uint8)
DMG_CMAP = np.array([Undefined, Not_dmg, Concrete_dmg, Rebar_dmg], dtype=np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True,
                    help="Input directory, usually the directory of predictions.")
parser.add_argument("--output", type=str, required=True,
                    help="Output directory if there is any output")
parser.add_argument("--raw_input", type=str,
                    help="The directory of original images.")
parser.add_argument("--cmp", action='store_true')
args = parser.parse_args()

def labelViz(img, num_class, cmap):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,), dtype=np.uint8)
    for i in range(num_class):
        img_out[img == i, :] = cmap[i]
    return img_out

def main():
    in_dir = args.input
    out_dir = args.output
    ori_dir = args.raw_input
    is_cmp = args.cmp

    img_list = os.listdir(in_dir)
    for img_name in img_list:
        img = cv2.imread(os.path.join(in_dir, img_name))
        if is_cmp:
            viz = labelViz(img, 9, CMP_CMAP)
        else:
            viz = labelViz(img, 4, DMG_CMAP)
        if not os.path.exists(os.path.join(out_dir, 'png')):
            os.makedirs(os.path.join(out_dir, 'png'))
        cv2.imwrite(os.path.join(out_dir, 'png' ,img_name.replace(".bmp", ".png")), viz)

        if not os.path.exists(os.path.join(out_dir, 'ori')):
            os.makedirs(os.path.join(out_dir, 'ori'))
        img = cv2.imread(os.path.join(ori_dir, img_name.replace(".bmp", "_Scene.png")))
        cv2.imwrite(os.path.join(out_dir, 'ori' ,img_name.replace(".bmp", "_Scene.png")), img)

if __name__ == "__main__":
    main()

    