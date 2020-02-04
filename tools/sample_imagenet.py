import shutil
import os
import os.path as osp
import glob
import numpy
import random
from concurrent import futures

raw_dir = "./CLS-LOC/train"
new_dir = "./Sample_imagenet/"
sample_ratio = 0.1
num_threads = 20
sample_base = int(1/sample_ratio)


def process(i, sub_cls):
    print ("Process cls {} name {}".format(i, sub_cls))
    sub_dir = osp.join(raw_dir, sub_cls)
    all_imgs = os.listdir(sub_dir)
    random.shuffle(all_imgs)
 
    ### create new dir
    sub_train_dir = osp.join(new_dir, 'train', sub_cls)
    sub_val_dir = osp.join(new_dir, 'valid', sub_cls)
    try:
        os.makedirs(sub_train_dir)
    except:
        pass
    try:
        os.makedirs(sub_val_dir)
    except:
        pass
    for j, img in enumerate(all_imgs):
        img_name = osp.join(sub_dir, img)

        if j%sample_base == 0:
            train_img_path = osp.join(sub_train_dir, img)
            shutil.copy(img_name, train_img_path)
        elif j%sample_base == 1:
            val_img_path = osp.join(sub_val_dir, img)
            shutil.copy(img_name, val_img_path)
        else:
            pass

if __name__ == '__main__':
    all_cls = os.listdir(raw_dir)
    random.shuffle(all_cls)

    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(process, i, sub_cls) for i, sub_cls in enumerate(all_cls)]
