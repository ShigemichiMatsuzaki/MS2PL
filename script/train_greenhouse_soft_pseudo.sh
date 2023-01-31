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

# Parameters
conf_thresh=0.95
entropy_loss_weight=1.0
kld_loss_weight=0.21879
label_update_epoch=5
label_weight_temp=3.536898
optimizer_name=SGD
scheduler_name=cyclic
label_weight_threshold=0.1
use_kld_class_loss=false
use_label_ent_weight=true
is_hard=false
python train_pseudo.py \
    --device cuda \
    --generate-pseudo-labels true \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --pseudo-label-batch-size 16 \
    --use-domain-gap true \
    --is-softmax-normalize true \
    --is-per-sample true \
    --is-per-pixel false \
    --resume-from ${RESUME_FROM} \
    --sp-label-min-portion 0.9 \
    --pseudo-label-save-path ./pseudo_labels/${camvid_model}/ \
    --target ${TARGET} \
    --train-data-list-path dataset/data_list/train_greenhouse_a.lst \
    --val-data-list-path dataset/data_list/val_greenhouse_a.lst \
    --test-data-list-path dataset/data_list/test_greenhouse_a.lst \
    --model ${MODEL} \
    --use-cosine true \
    --batch-size 64 \
    --epoch 30 \
    --lr 0.009 \
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
