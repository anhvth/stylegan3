
IMGID=$1
# TARGET=/shared/Data_HonTre/symlink_videos/003000/face_only/images/$IMGID.jpg
DIR=$2
TARGET=$DIR/$IMGID
# TARGET$1
# echo $TARGET
echo "%run" projector.py --network training-runs/00008-stylegan2--gpus6-batch24-gamma6.6/network-snapshot-003767.pkl --target $TARGET --outdir results/projected/$IMGID --num-steps 1000

# DIR="/data/Data_HonTre/symlink_videos/000334/cropped_face/track_000001/"
# paths = glob(f'{DIR}*')
# for path in paths[0::len(paths)//10]:
#     name = osp.basename(path)
#     !sh ./scripts/project.sh {name} {DIR}
