# MODEL=deeplabv3_mobilenet_v3_large
MODEL=espnetv2

python train.py \
    --device cuda \
    --model ${MODEL} \
    --batch-size 24 \
    --epoch 500 \
    --lr 0.0009 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler cyclic \
    --s1-name cityscapes
