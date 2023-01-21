MODEL=espnetv2
SOURCE_MODEL=espnetv2
# RESUME_FROM="/tmp/runs/domain_gap/camvid/espnetv2/20221123-214211/espnetv2_camvid_best_iou.pth"
TARGET=sakaki
if [ ${TARGET} = "greenhouse" ]; then
IGNORE_INDEX=3
elif [ ${TARGET} = "imo" ]; then
IGNORE_INDEX=3
elif [ ${TARGET} = "sakaki" ]; then
IGNORE_INDEX=5
fi
python train_pseudo.py \
    --device cuda \
    # Pseudo-label
    --generate-pseudo-labels false \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --pseudo-label-batch-size 16 \
    --is-hard false \
    --use-domain-gap true \
    --is-softmax-normalize true \
    --is-per-sample true \
    --is-per-pixel true \
    --sp-label-min-portion 0.9 \
    --pseudo-label-save-path ./pseudo_labels/${camvid_model}/ \
    # Training
    --model ${MODEL} \
    --target greenhouse \
    --train-data-list-path dataset/data_list/train_greenhouse_20230119.lst \
    --val-data-list-path dataset/data_list/val_greenhouse_a.lst \
    --batch-size 64 \
    --epoch 30 \
    --lr 0.009 \
    --label-update-epoch 20 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler constant \
    --class-wts-type uniform \
    --is-hard true \
    --is-old-label false \
    --use-prototype-denoising true \
    --label-weight-temperature 2.0 \
    --sp-label-min-portion 0.9 \
    --conf-thresh 0.95 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/${TARGET} \
    --ignore-index ${IGNORE_INDEX} 