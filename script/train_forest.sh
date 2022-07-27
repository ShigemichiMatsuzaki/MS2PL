python train.py \
    --device cuda \
    --model deeplabv3_mobilenet_v3_large \
    --batch-size 40 \
    --epoch 500 \
    --lr 0.0009 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler cyclic \
    --s1-name forest


#    --resume-from /tmp/runs/domain_gap/forest/20220622-093259/forest_ent_current.pth \
#    --resume-epoch 73 \