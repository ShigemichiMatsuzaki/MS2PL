python train.py \
    --device cuda \
    --model deeplabv3_mobilenet_v3_large \
    --batch-size 36 \
    --epoch 500 \
    --lr 0.0009 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler cyclic \
    --s1-name camvid