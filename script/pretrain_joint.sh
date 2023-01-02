MODEL=espnetv2
if [ ${MODEL} == "espnetv2" ]; then
BATCH_SIZE=40
elif [ ${MODEL} == "deeplabv3_resnet101" ]; then
BATCH_SIZE=4
else 
BATCH_SIZE=24
fi

IGNORE_INDEX=3

python train.py \
    --device cuda \
    --model ${MODEL} \
    --batch-size $BATCH_SIZE \
    --epoch 100 \
    --lr 0.009 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler exponential \
    --use-lr-warmup true \
    --weight-loss-ent 0.0 \
    --class-wts-type uniform \
    --train-image-size-h 256 \
    --train-image-size-w 480 \
    --val-image-size-h 256 \
    --val-image-size-w 480 \
    --ignore-index ${IGNORE_INDEX} \
    --use-other-datasets false \
    --use-label-conversion true \
    --use-cosine true \
    --s1-name camvid cityscapes forest