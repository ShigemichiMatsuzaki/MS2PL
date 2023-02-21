MODEL=espnetv2
SOURCE_MODEL=espnetv2
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

# Pseudo-label
camvid_model=${MODEL}
cityscapes_model=${MODEL}
forest_model=${MODEL}
camvid_weight="./pretrained_weights/espdnetue_2.0_480_best_camvid.pth"
cityscapes_weight="./pretrained_weights/espdnetue_2.0_512_best_city.pth"
forest_weight="./pretrained_weights/espdnetue_2.0_480_best_forest.pth"
optimizer_name=SGD
scheduler_name=exponential

# Parameters
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
    --epoch 30 \
    --lr 0.009 \
    --save-path /tmp/runs/domain_gap/ \
    --optim ${optimizer_name} \
    --scheduler ${scheduler_name} \
    --class-wts-type normal \
    --pseudo-label-dir ./pseudo_labels/MSDA_CL/${TARGET} \
    --ignore-index ${IGNORE_INDEX}
#    --optuna-resume-from ./pseudo_soft_espnetv2_20230103-111856.db
