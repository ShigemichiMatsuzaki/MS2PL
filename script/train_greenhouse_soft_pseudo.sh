MODEL=espnetv2
SOURCE_MODEL=espnetv2
RESUME_FROM=./pretrained_weights/espnetv2_camvid_cityscapes_forest_best_iou_norm.pth
# Parameters
label_weight_temp=5 
label_weight_threshold=0.1
use_kld_class_loss=false
use_label_ent_weight=true
is_hard=false
conf_thresh=0.95
python train_pseudo.py \
    --device cuda \
    --train-data-list-path dataset/data_list/train_greenhouse_a.lst \
    --val-data-list-path dataset/data_list/val_greenhouse_a.lst \
    --model ${MODEL} \
    --use-cosine true \
    --resume-from ${RESUME_FROM} \
    --target greenhouse \
    --batch-size 40 \
    --epoch 50 \
    --lr 0.009 \
    --label-update-epoch 1 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler constant \
    --class-wts-type uniform \
    --is-hard ${is_hard} \
    --use-kld-class-loss ${use_kld_class_loss} \
    --use-label-ent-weight ${use_label_ent_weight} \
    --conf-thresh ${conf_thresh} \
    --use-prototype-denoising true \
    --label-weight-temperature ${label_weight_temp} \
    --label-weight-threshold ${label_weight_threshold} \
    --kld-loss-weight 0.2 \
    --entropy-loss-weight 0.5 \
    --sp-label-min-portion 0.9 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/ \
    --ignore-index 3 