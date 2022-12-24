MODEL=espnetv2
SOURCE_MODEL=espnetv2

# Parameters
label_weight_temp=0.0 
label_weight_threshold=0.1

# 5 was the best
#for label_weight_temp in 1 2 3 4 5 10 20 30; do
#for label_weight_threshold in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
python train_pseudo.py \
    --device cuda \
    --model ${MODEL} \
    --target greenhouse \
    --batch-size 64 \
    --epoch 50 \
    --lr 0.009 \
    --label-update-epoch 5 \
    --conf-thresh 0.80 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler constant \
    --class-wts-type uniform \
    --is-hard false \
    --use-label-ent-weight true \
    --use-prototype-denoising false \
    --label-weight-temperature ${label_weight_temp} \
    --label-weight-threshold ${label_weight_threshold} \
    --kld-loss-weight 0.2 \
    --entropy-loss-weight 0.2 \
    --sp-label-min-portion 0.9 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/ \
    --ignore-index 3 
# done