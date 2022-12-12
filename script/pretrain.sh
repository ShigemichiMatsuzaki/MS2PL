# MODEL=deeplabv3_mobilenet_v3_large
# MODEL=espnetv2

DATASET=$1
MODEL=espnetv2
if [ ${MODEL} == "espnetv2" ]; then
BATCH_SIZE=40
elif [ ${MODEL} == "deeplabv3_resnet101" ]; then
BATCH_SIZE=4
else 
BATCH_SIZE=24
fi

if [ ${DATASET} == "camvid" ]; then
IGNORE_INDEX=12
elif [ ${DATASET} == "cityscapes" ]; then
IGNORE_INDEX=255
elif [ ${DATASET} == "forest" ]; then
IGNORE_INDEX=255
elif [ ${DATASET} == "gta5" ]; then
IGNORE_INDEX=255
fi

echo "$DATASET"

if [ -n "$DATASET" ];then
python train.py \
    --device cuda \
    --model ${MODEL} \
    --batch-size $BATCH_SIZE \
    --epoch 500 \
    --lr 0.009 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler polynomial \
    --use-lr-warmup true \
    --weight-loss-ent 0.0 \
    --class-wts-type inverse \
    --train-image-size-h 256 \
    --train-image-size-w 480 \
    --val-image-size-h 256 \
    --val-image-size-w 480 \
    --ignore-index ${IGNORE_INDEX} \
    --s1-name ${DATASET}
fi