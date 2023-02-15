MODEL=espnetv2
SOURCE_MODEL=espnetv2
RESUME_FROM=./pretrained_weights/espnetv2_camvid_cityscapes_forest_best_iou_norm.pth
USE_OPTUNA=false
# RESUME_FROM=""
TARGET=greenhouse
if [ ${TARGET} = "greenhouse" ]; then
IGNORE_INDEX=3
elif [ ${TARGET} = "imo" ]; then
IGNORE_INDEX=3
elif [ ${TARGET} = "sakaki" ]; then
IGNORE_INDEX=5
fi

# Pseudo-label
camvid_model=${MODEL}
cityscapes_model=${MODEL}
forest_model=${MODEL}
camvid_weight="./pretrained_weights/espdnetue_2.0_480_best_camvid.pth"
cityscapes_weight="./pretrained_weights/espdnetue_2.0_512_best_city.pth"
forest_weight="./pretrained_weights/espdnetue_2.0_480_best_forest.pth"

# Training Parameters
conf_thresh=0.95
entropy_loss_weight=1.0
kld_loss_weight=0.2
label_update_epoch=20
label_weight_temp=3.5
optimizer_name=SGD
scheduler_name=constant
label_weight_threshold=0.1
use_kld_class_loss=false
use_label_ent_weight=true
is_hard=false
# Pseudo-label parameters
use_domain_gap=true
is_per_sample=true
is_per_pixel=false
is_softmax_normalize=true
python train_pseudo.py \
    --device cuda \
    --generate-pseudo-labels true \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --pseudo-label-batch-size 16 \
    --use-domain-gap ${use_domain_gap} \
    --is-per-sample ${is_per_sample} \
    --is-per-pixel ${is_per_pixel} \
    --is-softmax-normalize $is_softmax_normalize \
    --sp-label-min-portion 0.9 \
    --pseudo-label-save-path ./pseudo_labels/${camvid_model}/ \
    --target ${TARGET} \
    --train-data-list-path dataset/data_list/train_sakaki.lst \
    --val-data-list-path dataset/data_list/test_sakaki.lst \
    --test-data-list-path dataset/data_list/test_sakaki.lst \
    --model ${MODEL} \
    --use-cosine true \
    --batch-size 64 \
    --epoch 100 \
    --lr 0.015 \
    --label-update-epoch ${label_update_epoch} \
    --save-path /tmp/runs/domain_gap/ \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type normal \
    --is-hard ${is_hard} \
    --use-kld-class-loss ${use_kld_class_loss} \
    --use-label-ent-weight ${use_label_ent_weight} \
    --conf-thresh ${conf_thresh} \
    --use-prototype-denoising true \
    --label-weight-temperature ${label_weight_temp} \
    --kld-loss-weight ${kld_loss_weight} \
    --entropy-loss-weight ${entropy_loss_weight} \
    --sp-label-min-portion 0.9 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/${TARGET} \
    --ignore-index ${IGNORE_INDEX} \
    --use-optuna ${USE_OPTUNA} 
#    --optuna-resume-from ./pseudo_soft_espnetv2_20230103-111856.db
