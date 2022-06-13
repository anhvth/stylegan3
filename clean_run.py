import os
import os.path as osp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input')
args = parser.parse_args()



from glob import glob
paths = glob(osp.join(args.input, '*'))


for directory in paths:
    paths = glob(osp.join(directory, '*.pkl'))
    max_num = 0
    for path in paths:
        name = osp.basename(path)
        num = int(''.join(_ for _ in name if _.isdigit()))
        max_num = num if num > max_num else max_num
    if max_num < 1000:
        os.system(f'rm -r {directory}')
    print(directory, max_num)