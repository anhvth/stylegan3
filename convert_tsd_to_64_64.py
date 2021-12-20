from avcv.utils import *
import os.path as osp
from glob import glob
import mmcv
import numpy as np
out_dir = './data/tsd_cropted/'

# paths = glob("/data/tsd/classification/GTSRB/**/*.png", recursive=True)
# paths = glob("/data/tsd/classification/cropped/*json", recursive=True)

# meta = dict()
# for path in paths:
#     meta.update(mmcv.load(path)['annotations'])
# import pandas as pd
# df = pd.DataFrame.from_dict(meta, orient='index')
from PIL import Image
# df['size'] = df.path

paths = glob("/data/tsd/classification/cropped/**/*.jpg", recursive=True)
sizes = [Image.open(p) for p in paths]
import pandas as pd
df = pd.DataFrame(data=paths, columns=['path'])
df['size'] = df.path.apply(lambda x: Image.open(x).size)


def f(path):
    img = mmcv.imread(path)
    h, w = img.shape[:2]
    ratio = h/w
    if ratio > 1.3 or ratio < 0.7:
        return
    img = mmcv.imresize(img, (64, 64))
    out_path = osp.join(out_dir, str(np.random.choice(1000)),osp.basename(path))
    mmcv.imwrite(img, out_path)

from multiprocessing import Pool
import tqdm
with Pool() as p:
    r = list(tqdm.tqdm(p.imap(f, paths), total=len(paths)))

# python train.py --outdir=./training-runs --data=data/dest_gtsr --gpus=1 --dry-run