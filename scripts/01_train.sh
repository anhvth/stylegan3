# python train.py --outdir=./training-runs --cfg=stylegan2 --data=data/tsr_64_dest --gpus=8 --batch=64 --gamma=8.2
python train.py --outdir=./training-runs --cfg=stylegan2 --data=data/gtsr_64_dest --gpus=8 --batch=128 --gamma=8.2 --cond=1
# python train.py --outdir=./training-runs --cfg=stylegan3-t --data=data/tsr_64_dest --gpus=1 --batch=4 --gamma=8.2

