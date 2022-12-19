MODEL=espnetv2
SOURCE_MODEL=espnetv2
# RESUME_FROM="/tmp/runs/domain_gap/camvid/espnetv2/20221123-214211/espnetv2_camvid_best_iou.pth"
RESUME_FROM=./pretrained_weights/espdnetue_2.0_480_best_camvid.pth
python train_pseudo.py \
    --device cuda \
    --model ${MODEL} \
    --target greenhouse \
    --batch-size 64 \
    --epoch 30 \
    --lr 0.009 \
    --label-update-epoch 5 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler constant \
    --class-wts-type uniform \
    --is-hard true \
    --is-old-label false \
    --use-prototype-denoising true \
    --label-weight-temperature 2.0 \
    --sp-label-min-portion 0.9 \
    --conf-thresh 0.95 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/ \
    --ignore-index 3 