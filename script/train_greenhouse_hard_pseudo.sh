MODEL=espnetv2
SOURCE_MODEL=espnetv2
RESUME_FROM="/tmp/runs/domain_gap/camvid/espnetv2/20221123-214211/espnetv2_camvid_best_iou.pth"
# RESUME_FROM=/tmp/runs/domain_gap/camvid/unet/20221021-132337/unet_camvid_best_iou.pth
python train_pseudo.py \
    --device cuda \
    --model ${MODEL} \
    --target greenhouse \
    --batch-size 48 \
    --epoch 30 \
    --lr 0.0090 \
    --label-update-epoch 10 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler constant \
    --class-wts-type normal \
    --is-hard true \
    --is-old-label false \
    --conf-thresh 0.95 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/ \
    --ignore-index 3 