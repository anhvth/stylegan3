# from label_tool_utils import get_preloaded_anns, get_unlabeled_ann_ids
from multiprocessing import Pool
from tqdm import tqdm
from glob import glob
import mmcv
import numpy as np
import os.path as osp

VIET_MAP_SUPER_CLASS_SET = {'Bien cam', 'Hieu Lenh', 'Nguy Hiem'}
out_dir = './data/tsr_64'
meta_paths = glob('/data/tsd/classification/cropped/*.json')
meta = dict()

for path in meta_paths:
    meta.update(mmcv.load(path)['annotations'])


def f(v):
    if not v['super_class_name'] in VIET_MAP_SUPER_CLASS_SET:
        return
    path = v['file_path']
    img = mmcv.imread(path)
    h, w = img.shape[:2]
    ratio = h/w
    if ratio > 1.5 or ratio < 0.5:
        return
    img = mmcv.imresize(img, (64, 64))
    out_path = osp.join(out_dir, "{:06d}".format(np.random.choice(1000)),
                        osp.basename(path))
    mmcv.imwrite(img, out_path)

values = list(meta.values())
with Pool(None) as p:
    r = list(tqdm(p.imap(f, values), total=len(values)))
