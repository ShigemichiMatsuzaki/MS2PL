MODEL=deeplabv3_mobilenet_v3_large
SOURCE_MODEL=deeplabv3_mobilenet_v3_large
python train_pseudo.py \
    --device cuda \
    --model ${MODEL} \
    --target greenhouse \
    --batch-size 64 \
    --epoch 30 \
    --lr 0.0009 \
    --label-update-epoch 10 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler cyclic \
    --is-hard false \
    --conf-thresh 0.6 \
    --use-label-ent-weight true \
    --label-weight-temperature 2.0 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/ \
    --ignore-index 3 
