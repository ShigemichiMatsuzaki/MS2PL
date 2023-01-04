MODEL=espnetv2
SOURCE_MODEL=espnetv2
# RESUME_FROM=./pretrained_weights/espnetv2_camvid_cityscapes_forest_best_iou_norm.pth
RESUME_FROM=""

# Parameters
conf_thresh=0.95
entropy_loss_weight=1.0
kld_loss_weight=0.21879
label_update_epoch=11
label_weight_temp=3.536898
optimizer_name=SGD
scheduler_name=constant
label_weight_threshold=0.1
use_kld_class_loss=false
use_label_ent_weight=true
is_hard=false
python train_pseudo.py \
    --device cuda \
    --train-data-list-path dataset/data_list/train_greenhouse_a.lst \
    --val-data-list-path dataset/data_list/val_greenhouse_a.lst \
    --model ${MODEL} \
    --use-cosine true \
    --target greenhouse \
    --batch-size 128 \
    --epoch 20 \
    --lr 0.019 \
    --label-update-epoch 1 \
    --save-path /tmp/runs/domain_gap/ \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type uniform \
    --use-optuna true \
    --is-hard ${is_hard} \
    --use-kld-class-loss ${use_kld_class_loss} \
    --use-label-ent-weight ${use_label_ent_weight} \
    --conf-thresh ${conf_thresh} \
    --use-prototype-denoising true \
    --label-weight-temperature ${label_weight_temp} \
    --kld-loss-weight ${kld_loss_weight} \
    --entropy-loss-weight ${entropy_loss_weight} \
    --sp-label-min-portion 0.9 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/ \
    --ignore-index 3 