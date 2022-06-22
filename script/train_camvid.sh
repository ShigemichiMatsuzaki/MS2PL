python train.py \
    --device cuda \
    --batch-size 16 \
    --lr 0.00005 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler cyclic \
    --s1-name camvid