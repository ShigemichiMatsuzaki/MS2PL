#cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220723-154300/deeplabv3_mobilenet_v3_large_cityscapes_best_iou.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220728-160250/deeplabv3_mobilenet_v3_large_forest_best_iou.pth"
# camvid_weight="/tmp/runs/domain_gap/camvid/20220801-042638/deeplabv3_mobilenet_v3_large_camvid_best_iou.pth"

model="espnetv2"
camvid_model=${model}
cityscapes_model=${model}
forest_model=${model}

model_type="best_iou"

# DeepLab-MobileNet
if [ ${model} = "unet" ]; then
    # UNet
    camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221010-201507/${camvid_model}_camvid_${model_type}.pth"
    #camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221001-213201/${camvid_model}_camvid_best_ent_loss.pth"
    cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221006-232113/${cityscapes_model}_cityscapes_${model_type}.pth"
    forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221011-180741/${forest_model}_forest_${model_type}.pth"
    # No ent loss
    camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221031-180408/${camvid_model}_camvid_${model_type}.pth"
    cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221031-012334/${cityscapes_model}_cityscapes_${model_type}.pth"
    # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221022-122052/${forest_model}_forest_${model_type}.pth"
    forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221027-232513/${forest_model}_forest_${model_type}.pth"

elif [ ${model} = "espnetv2" ]; then
    camvid_weight="./pretrained_weights/espdnetue_2.0_480_best_camvid.pth"
    cityscapes_weight="./pretrained_weights/espdnetue_2.0_512_best_city.pth"
    forest_weight="./pretrained_weights/espdnetue_2.0_480_best_forest.pth"
    # Inverse
    # camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221123-060632/${camvid_model}_camvid_${model_type}.pth"
    # cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221122-233111/${cityscapes_model}_cityscapes_${model_type}.pth"
    # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221123-074219/${forest_model}_forest_${model_type}.pth"
    # Normal
    # camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221123-214211/${camvid_model}_camvid_${model_type}.pth"
    # cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221124-003304/${cityscapes_model}_cityscapes_${model_type}.pth"
    # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221123-230959/${forest_model}_forest_${model_type}.pth"
elif [ ${model} = "deeplabv3_mobilenet_v3_large" ]; then
    camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20220725-034846/${camvid_model}_camvid_best_iou.pth"
    cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20220801-233634/${cityscapes_model}_cityscapes_best_iou.pth"
    forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20220728-160250/${forest_model}_forest_best_iou.pth"
fi

TARGET=imo
if [ ${TARGET} = "greenhouse" ]; then
TRAIN_LST=train_greenhouse_a.lst
VAL_LST=val_greenhouse_a.lst
TEST_LST=test_greenhouse_a.lst
IGNORE_INDEX=3
elif [ ${TARGET} = "imo" ]; then
IGNORE_INDEX=3
TRAIN_LST=train_imo_stabilized.lst
VAL_LST=test_imo.lst
TEST_LST=test_imo.lst
elif [ ${TARGET} = "sakaki" ]; then
IGNORE_INDEX=5
TRAIN_LST=train_sakaki.lst
VAL_LST=test_sakaki.lst
TEST_LST=test_sakaki.lst
fi

python evaluate_source_models.py \
    --device cuda \
    --target ${TARGET} \
    --target-data-list ./dataset/data_list/vis_imo.lst \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --save-path ./pseudo_labels/
