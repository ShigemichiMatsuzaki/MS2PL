MODEL=espnetv2
if [ ${MODEL} == "espnetv2" ]; then
BATCH_SIZE=64
elif [ ${MODEL} == "deeplabv3_resnet101" ]; then
BATCH_SIZE=4
else 
BATCH_SIZE=24
fi

TARGET=sakaki
if [ ${TARGET} = "greenhouse" ]; then
IGNORE_INDEX=3
elif [ ${TARGET} = "imo" ]; then
IGNORE_INDEX=3
elif [ ${TARGET} = "sakaki" ]; then
IGNORE_INDEX=5
fi

python train.py \
    --device cuda \
    --model ${MODEL} \
    --batch-size $BATCH_SIZE \
    --target ${TARGET} \
    --epoch 100 \
    --lr 0.009 \
    --lr-gamma 0.9 \
    --save-path /tmp/runs/domain_gap/${TARGET} \
    --scheduler exponential \
    --use-lr-warmup false \
    --weight-loss-ent 0.0 \
    --class-wts-type uniform \
    --train-image-size-h 256 \
    --train-image-size-w 480 \
    --val-image-size-h 256 \
    --val-image-size-w 480 \
    --ignore-index ${IGNORE_INDEX} \
    --use-other-datasets false \
    --use-cosine true \
    --s1-name camvid cityscapes forest