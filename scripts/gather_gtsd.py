# from label_tool_utils import get_preloaded_anns, get_unlabeled_ann_ids
from multiprocessing import Pool
from tqdm import tqdm
from glob import glob
import mmcv
import numpy as np
import os.path as osp

# VIET_MAP_SUPER_CLASS_SET = {'Bien cam', 'Hieu Lenh', 'Nguy Hiem'}
out_dir = './data/gtsr_64'
paths = glob('/data/gtsd/train/*/*.png')
meta = dict()

def f(path):
    img = mmcv.imread(path)
    h, w = img.shape[:2]
    ratio = h/w
    if ratio > 1.5 or ratio < 0.5:
        return
    img = mmcv.imresize(img, (64, 64))
    out_path = osp.join(out_dir, path.split('/')[-2],
                        osp.basename(path))
    mmcv.imwrite(img, out_path)

with Pool(None) as p:
    r = list(tqdm(p.imap(f, paths), total=len(paths)))
