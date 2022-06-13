DATA="data/"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,7
python train.py --metrics none  --cbase 16384 --data $DATA --gpus=6 --outdir=./training-runs --cfg=stylegan2 --batch=24 --gamma=6.6 --mirror=1 --kimg=5000 --snap=20 $@