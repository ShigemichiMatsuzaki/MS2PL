#cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220723-154300/deeplabv3_mobilenet_v3_large_cityscapes_best_iou.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220728-160250/deeplabv3_mobilenet_v3_large_forest_best_iou.pth"
# camvid_weight="/tmp/runs/domain_gap/camvid/20220801-042638/deeplabv3_mobilenet_v3_large_camvid_best_iou.pth"

camvid_model="unet"
cityscapes_model="unet"
forest_model="unet"
model_type="best_ent_loss"

# DeepLab-MobileNet
camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20220725-034846/${camvid_model}_camvid_best_iou.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20220801-233634/${cityscapes_model}_cityscapes_best_iou.pth"
forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20220728-160250/${forest_model}_forest_best_iou.pth"

# ESPNetv2
camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221001-213201/${camvid_model}_camvid_${model_type}.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221005-203157/${cityscapes_model}_cityscapes_${model_type}.pth"
forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221004-104824/${forest_model}_forest_${model_type}.pth"

# UNet
camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221010-201507/${camvid_model}_camvid_${model_type}.pth"
#camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221001-213201/${camvid_model}_camvid_best_ent_loss.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221006-232113/${cityscapes_model}_cityscapes_${model_type}.pth"
forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221011-180741/${forest_model}_forest_${model_type}.pth"

python evaluate_source_models.py \
    --device cuda \
    --target greenhouse \
    --target-data-list ./dataset/data_list/val_greenhouse_a.lst \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --save-path ./pseudo_labels/
