MODEL=espnetv2
SOURCE_MODEL=espnetv2
# RESUME_FROM=""
TARGET=sakaki
if [ ${TARGET} = "greenhouse" ]; then
    TEST_LST=test_greenhouse_a.lst
    RESUME_FROM=./pretrained_weights/espnetv2_camvid_cityscapes_forest_best_iou_norm.pth
elif [ ${TARGET} = "imo" ]; then
    TEST_LST=test_sakaki.lst
    RESUME_FROM="./pretrained_weights/espnetv2_camvid_cityscapes_forest_best_iou_sakaki.pth"
elif [ ${TARGET} = "sakaki" ]; then
    TEST_LST=test_sakaki.lst
    RESUME_FROM="./pretrained_weights/espnetv2_camvid_cityscapes_forest_best_iou_sakaki.pth"
fi
SAVE_PATH=./results/

BATCH_SIZE=128
python test_single_model.py \
    --device cuda \
    --model ${MODEL} \
    --resume-from ${RESUME_FROM} \
    --batch-size ${BATCH_SIZE} \
    --target ${TARGET} \
    --test-data-list-path dataset/data_list/${TEST_LST} \
    --use-cosine true \
    --test-save-path ${SAVE_PATH}

# Pseudo-label
camvid_model=${MODEL}
cityscapes_model=${MODEL}
forest_model=${MODEL}
camvid_weight="./pretrained_weights/espdnetue_2.0_480_best_camvid.pth"
cityscapes_weight="./pretrained_weights/espdnetue_2.0_512_best_city.pth"
forest_weight="./pretrained_weights/espdnetue_2.0_480_best_forest.pth"
python test_ensemble.py \
    --device cuda \
    --model ${MODEL} \
    --resume-from ${RESUME_FROM} \
    --batch-size ${BATCH_SIZE} \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --target ${TARGET} \
    --test-data-list-path dataset/data_list/${TEST_LST} \
    --use-cosine true \
    --test-save-path ${SAVE_PATH}