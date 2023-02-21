MODEL=espnetv2
SOURCE_MODEL=espnetv2
RESUME_FROM=./pretrained_weights/espnetv2_camvid_cityscapes_forest_best_iou_norm.pth
# RESUME_FROM=""
TARGET=sakaki
if [ ${TARGET} = "greenhouse" ]; then
    IGNORE_INDEX=3
    TRAIN_LST=train_greenhouse_a.lst
    VAL_LST=val_greenhouse_a.lst
    TEST_LST=test_greenhouse_a.lst
elif [ ${TARGET} = "imo" ]; then
    IGNORE_INDEX=3
    TRAIN_LST=train_imo_stabilized.lst
    VAL_LST=test_sakaki.lst
    TEST_LST=test_sakaki.lst
elif [ ${TARGET} = "sakaki" ]; then
    IGNORE_INDEX=5
    TRAIN_LST=train_sakaki.lst
    VAL_LST=test_sakaki.lst
    TEST_LST=test_sakaki.lst
fi
SAVE_PATH=/tmp/runs/domain_gap/exp/comparative/
BATCH_SIZE=64
LEARNING_RATE=0.02

VAL_EVERY_EPOCHS=1
VIS_EVERY_VALS=5
EPOCH=50

# Pseudo-label
camvid_model=${MODEL}
cityscapes_model=${MODEL}
forest_model=${MODEL}
camvid_weight="./pretrained_weights/espdnetue_2.0_480_best_camvid.pth"
cityscapes_weight="./pretrained_weights/espdnetue_2.0_512_best_city.pth"
forest_weight="./pretrained_weights/espdnetue_2.0_480_best_forest.pth"

# Parameters
conf_thresh=0.95
entropy_loss_weight=0.5
kld_loss_weight=0.21879
label_update_epoch=20
label_weight_temp=3.536898
optimizer_name=SGD
scheduler_name=cyclic
label_weight_threshold=0.1
use_kld_class_loss=false
use_label_ent_weight=true
is_hard=false
# Triple soft
python train_pseudo.py \
    --device cuda \
    --generate-pseudo-labels true \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --pseudo-label-batch-size 16 \
    --domain-gap-type "per_sample" \
    --is-softmax-normalize true \
    --resume-from ${RESUME_FROM} \
    --sp-label-min-portion 0.9 \
    --target ${TARGET} \
    --train-data-list-path dataset/data_list/${TRAIN_LST} \
    --val-data-list-path dataset/data_list/${VAL_LST} \
    --test-data-list-path dataset/data_list/${TEST_LST} \
    --model ${MODEL} \
    --use-cosine true \
    --batch-size ${BATCH_SIZE} \
    --epoch ${EPOCH} \
    --lr ${LEARNING_RATE} \
    --val-every-epochs ${VAL_EVERY_EPOCHS} \
    --vis-every-vals ${VIS_EVERY_VALS} \
    --label-update-epoch ${label_update_epoch} \
    --save-path /tmp/runs/domain_gap/ \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type normal \
    --is-hard ${is_hard} \
    --use-kld-class-loss ${use_kld_class_loss} \
    --use-label-ent-weight ${use_label_ent_weight} \
    --is-sce-loss true \
    --sce-alpha 0.1 \
    --sce-beta 1.0 \
    --conf-thresh ${conf_thresh} \
    --use-prototype-denoising true \
    --label-weight-temperature ${label_weight_temp} \
    --kld-loss-weight ${kld_loss_weight} \
    --entropy-loss-weight ${entropy_loss_weight} \
    --sp-label-min-portion 0.9 \
    --initial-pseudo-label-path ./pseudo_labels/${SOURCE_MODEL}/${TARGET} \
    --ignore-index ${IGNORE_INDEX} \
    --use-optuna false
##    --optuna-resume-from ./pseudo_soft_espnetv2_20230103-111856.db
#
## Triple hard
python train_pseudo.py \
    --device cuda \
    --generate-pseudo-labels true \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --pseudo-label-batch-size 16 \
    --is-softmax-normalize true \
    --domain-gap-type "per_sample" \
    --sp-label-min-portion 0.9 \
    --target ${TARGET} \
    --train-data-list-path dataset/data_list/${TRAIN_LST} \
    --val-data-list-path dataset/data_list/${VAL_LST} \
    --test-data-list-path dataset/data_list/${TEST_LST} \
    --model ${MODEL} \
    --use-cosine true \
    --batch-size ${BATCH_SIZE} \
    --epoch ${EPOCH} \
    --lr ${LEARNING_RATE} \
    --val-every-epochs ${VAL_EVERY_EPOCHS} \
    --vis-every-vals ${VIS_EVERY_VALS} \
    --label-update-epoch ${label_update_epoch} \
    --save-path ${SAVE_PATH} \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type normal \
    --is-hard true \
    --use-kld-class-loss ${use_kld_class_loss} \
    --use-label-ent-weight ${use_label_ent_weight} \
    --is-sce-loss false \
    --conf-thresh ${conf_thresh} \
    --use-prototype-denoising true \
    --label-weight-temperature ${label_weight_temp} \
    --kld-loss-weight ${kld_loss_weight} \
    --entropy-loss-weight ${entropy_loss_weight} \
    --initial-pseudo-label-path ./pseudo_labels/${SOURCE_MODEL}/${TARGET} \
    --sp-label-min-portion 0.9 \
    --ignore-index ${IGNORE_INDEX} \
    --use-optuna false

# Triple MSDA_CL 
python train_msdacl.py \
    --device cuda \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --target ${TARGET} \
    --train-data-list-path dataset/data_list/${TRAIN_LST} \
    --val-data-list-path dataset/data_list/${VAL_LST} \
    --test-data-list-path dataset/data_list/${TEST_LST} \
    --model ${MODEL} \
    --use-cosine true \
    --batch-size 40 \
    --epoch ${EPOCH} \
    --lr ${LEARNING_RATE} \
    --save-path ${SAVE_PATH} \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type normal \
    --pseudo-label-dir ./pseudo_labels/msda_cl/${TARGET} \
    --ignore-index ${IGNORE_INDEX}
#    --optuna-resume-from ./pseudo_soft_espnetv2_20230103-111856.db

# Double (CS+FR) soft
python train_pseudo.py \
    --device cuda \
    --generate-pseudo-labels true \
    --source-model-names ${cityscapes_model},${forest_model} \
    --source-dataset-names cityscapes,forest \
    --source-weight-names ${cityscapes_weight},${forest_weight} \
    --pseudo-label-batch-size 16 \
    --is-softmax-normalize true \
    --domain-gap-type "per_sample" \
    --sp-label-min-portion 0.9 \
    --target ${TARGET} \
    --train-data-list-path dataset/data_list/${TRAIN_LST} \
    --val-data-list-path dataset/data_list/${VAL_LST} \
    --test-data-list-path dataset/data_list/${TEST_LST} \
    --model ${MODEL} \
    --use-cosine true \
    --batch-size ${BATCH_SIZE} \
    --epoch ${EPOCH} \
    --lr ${LEARNING_RATE} \
    --val-every-epochs ${VAL_EVERY_EPOCHS} \
    --vis-every-vals ${VIS_EVERY_VALS} \
    --label-update-epoch ${label_update_epoch} \
    --save-path ${SAVE_PATH} \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type normal \
    --is-hard false \
    --use-kld-class-loss ${use_kld_class_loss} \
    --use-label-ent-weight ${use_label_ent_weight} \
    --is-sce-loss true \
    --sce-alpha 0.1 \
    --sce-beta 1.0 \
    --conf-thresh ${conf_thresh} \
    --use-prototype-denoising true \
    --label-weight-temperature ${label_weight_temp} \
    --kld-loss-weight ${kld_loss_weight} \
    --entropy-loss-weight ${entropy_loss_weight} \
    --sp-label-min-portion 0.9 \
    --initial-pseudo-label-path ./pseudo_labels/${SOURCE_MODEL}/${TARGET} \
    --ignore-index ${IGNORE_INDEX} \
    --use-optuna false
#    --optuna-resume-from ./pseudo_soft_espnetv2_20230103-111856.db

# Double (CS+FR) hard
python train_pseudo.py \
    --device cuda \
    --generate-pseudo-labels true \
    --source-model-names ${cityscapes_model},${forest_model} \
    --source-dataset-names cityscapes,forest \
    --source-weight-names ${cityscapes_weight},${forest_weight} \
    --pseudo-label-batch-size 16 \
    --is-softmax-normalize true \
    --domain-gap-type "per_sample" \
    --sp-label-min-portion 0.9 \
    --target ${TARGET} \
    --train-data-list-path dataset/data_list/${TRAIN_LST} \
    --val-data-list-path dataset/data_list/${VAL_LST} \
    --test-data-list-path dataset/data_list/${TEST_LST} \
    --model ${MODEL} \
    --use-cosine true \
    --batch-size ${BATCH_SIZE} \
    --epoch ${EPOCH} \
    --lr ${LEARNING_RATE} \
    --val-every-epochs ${VAL_EVERY_EPOCHS} \
    --vis-every-vals ${VIS_EVERY_VALS} \
    --label-update-epoch ${label_update_epoch} \
    --save-path ${SAVE_PATH} \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type normal \
    --is-hard true \
    --use-kld-class-loss ${use_kld_class_loss} \
    --use-label-ent-weight ${use_label_ent_weight} \
    --is-sce-loss false \
    --conf-thresh ${conf_thresh} \
    --use-prototype-denoising true \
    --label-weight-temperature ${label_weight_temp} \
    --kld-loss-weight ${kld_loss_weight} \
    --entropy-loss-weight ${entropy_loss_weight} \
    --sp-label-min-portion 0.9 \
    --initial-pseudo-label-path ./pseudo_labels/${SOURCE_MODEL}/${TARGET} \
    --ignore-index ${IGNORE_INDEX} \
    --use-optuna false
#    --optuna-resume-from ./pseudo_soft_espnetv2_20230103-111856.db

# Double MSDA_CL 
python train_msdacl.py \
    --device cuda \
    --source-model-names ${cityscapes_model},${forest_model} \
    --source-dataset-names cityscapes,forest \
    --source-weight-names ${cityscapes_weight},${forest_weight} \
    --target ${TARGET} \
    --train-data-list-path dataset/data_list/${TRAIN_LST} \
    --val-data-list-path dataset/data_list/${VAL_LST} \
    --test-data-list-path dataset/data_list/${TEST_LST} \
    --model ${MODEL} \
    --use-cosine true \
    --batch-size 40 \
    --epoch ${EPOCH} \
    --lr ${LEARNING_RATE} \
    --save-path ${SAVE_PATH} \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type normal \
    --pseudo-label-dir ./pseudo_labels/msda_cl/${TARGET} \
    --ignore-index ${IGNORE_INDEX}
