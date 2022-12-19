model="espnetv2"
camvid_model=${model}
cityscapes_model=${model}
forest_model=${model}

model_type="best_iou"

# DeepLab-MobileNet
if [ ${model} = "unet" ]; then
    # UNet
    camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221010-201507/${camvid_model}_camvid_${model_type}.pth"
    cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221006-232113/${cityscapes_model}_cityscapes_${model_type}.pth"
    forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221011-180741/${forest_model}_forest_${model_type}.pth"
    # No ent loss
    camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221030-180408/${camvid_model}_camvid_${model_type}.pth"
    cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221031-012334/${cityscapes_model}_cityscapes_${model_type}.pth"
    # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221022-122052/${forest_model}_forest_${model_type}.pth"
    forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221027-232513/${forest_model}_forest_${model_type}.pth"
elif [ ${model} = "espnetv2" ]; then
    camvid_weight="./pretrained_weights/espdnetue_2.0_480_best_camvid.pth"
    cityscapes_weight="./pretrained_weights/espdnetue_2.0_512_best_city.pth"
    forest_weight="./pretrained_weights/espdnetue_2.0_480_best_forest.pth"
    # Uniform weight
    # camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221104-221219/${camvid_model}_camvid_${model_type}.pth"
    # cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221105-050515/${cityscapes_model}_cityscapes_${model_type}.pth"
    # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221104-141705/${forest_model}_forest_${model_type}.pth"

    # # Inverse
    # camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221124-123644/${camvid_model}_camvid_${model_type}.pth"
    # cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221124-171652/${cityscapes_model}_cityscapes_${model_type}.pth"
    # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221124-145751/${forest_model}_forest_${model_type}.pth"
    # # Normal
    # # camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221122-130747/${camvid_model}_camvid_${model_type}.pth"
    # # cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221122-164200/${cityscapes_model}_cityscapes_${model_type}.pth"
    # # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221122-150258/${forest_model}_forest_${model_type}.pth"
    # camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221123-214211/${camvid_model}_camvid_${model_type}.pth"
    # cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221124-003304/${cityscapes_model}_cityscapes_${model_type}.pth"
    # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221123-230959/${forest_model}_forest_${model_type}.pth"
elif [ ${model} = "deeplabv3_mobilenet_v3_large" ]; then
    camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20220725-034846/${camvid_model}_camvid_best_iou.pth"
    cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20220801-233634/${cityscapes_model}_cityscapes_best_iou.pth"
    forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20220728-160250/${forest_model}_forest_best_iou.pth"
fi

python generate_pseudo_labels.py \
    --device cuda \
    --target-data-list ./dataset/data_list/train_greenhouse_a.lst \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --batch-size 12 \
    --is-hard false \
    --is-softmax-normalize true \
    --is-per-sample false \
    --is-per-pixel false \
    --sp-label-min-portion 0.9 \
    --save-path ./pseudo_labels/${camvid_model}/
