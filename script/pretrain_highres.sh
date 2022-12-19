DATASET=$1
MODEL=deeplabv3_resnet50
BATCH_SIZE=6 # ResNet50 

if [ ${DATASET} == "cityscapes" ]; then
IGNORE_INDEX=255
elif [ ${DATASET} == "gta5" ]; then
IGNORE_INDEX=255
fi


if [ -n "$DATASET" ];then
python train.py \
    --device cuda \
    --model ${MODEL} \
    --batch-size ${BATCH_SIZE} \
    --epoch 300 \
    --lr 0.0009 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler polynomial \
    --weight-loss-ent 0.0 \
    --class-wts-type normal \
    --train-image-size-h 512 \
    --train-image-size-w 1024 \
    --val-image-size-h 512 \
    --val-image-size-w 1024 \
    --ignore-index ${IGNORE_INDEX} \
    --s1-name ${DATASET}
fi