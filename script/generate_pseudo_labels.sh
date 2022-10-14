camvid_model="unet"
cityscapes_model="unet"
forest_model="unet"
model_type="best_iou"

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

python generate_pseudo_labels.py \
    --device cuda \
    --target-data-list ./dataset/data_list/train_greenhouse_a.lst \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --batch-size 32 \
    --is-hard true \
    --is-softmax-normalize false \
    --superpixel-pseudo-min-portion 0.7 \
    --save-path ./pseudo_labels/${camvid_model}/
