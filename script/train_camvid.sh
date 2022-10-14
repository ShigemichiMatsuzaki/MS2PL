# MODEL=deeplabv3_mobilenet_v3_large
# MODEL=espnetv2
MODEL=unet

python train.py \
    --device cuda \
    --model ${MODEL} \
    --batch-size 16 \
    --epoch 500 \
    --lr 0.0009 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler cyclic \
    --s1-name camvid