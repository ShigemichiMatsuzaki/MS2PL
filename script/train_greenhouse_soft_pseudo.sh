MODEL=espnetv2
SOURCE_MODEL=espnetv2
python train_pseudo.py \
    --device cuda \
    --model ${MODEL} \
    --target greenhouse \
    --batch-size 24 \
    --epoch 30 \
    --lr 0.0009 \
    --label-update-epoch 10 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler cyclic \
    --is-hard false \
    --conf-thresh 0.90 \
    --use-label-ent-weight true \
    --label-weight-temperature 2.0 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/ \
    --ignore-index 3 
