#camvid_weight="/tmp/runs/domain_gap/camvid/20220614-094914/camvid_ent_current.pth"
#camvid_weight="/tmp/runs/domain_gap/camvid/20220610-121207/camvid_ent_best.pth"
#camvid_weight="/tmp/runs/domain_gap/camvid/20220625-120300/camvid_ent_current.pth"
#cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220611-115709/cityscapes_ent_best.pth"
#cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220619-220126/cityscapes_ent_current.pth"
#cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220626-202624/cityscapes_ent_current.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220612-133456/forest_ent_best.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220622-174736/forest_ent_current.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220628-154333/forest_ent_current.pth"
# Previous (7/26)
# camvid_weight="/tmp/runs/domain_gap/camvid/20220630-113848/camvid_ent_current.pth"
# cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220630-222203/cityscapes_ent_current.pth"
# forest_weight="/tmp/runs/domain_gap/forest/20220701-231435/forest_ent_current.pth"
# New (7/26)

# cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220723-154300/deeplabv3_mobilenet_v3_large_cityscapes_best_ent_loss.pth"
# forest_weight="/tmp/runs/domain_gap/forest/20220722-155508/deeplabv3_mobilenet_v3_large_forest_best_ent_loss.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220727-220126/deeplabv3_mobilenet_v3_large_forest_best_ent_loss.pth"
# forest_weight="/tmp/runs/domain_gap/forest/20220728-160250/deeplabv3_mobilenet_v3_large_forest_best_ent_loss.pth"
# New (8/2)
# camvid_weight="/tmp/runs/domain_gap/camvid/20220801-042638/deeplabv3_mobilenet_v3_large_camvid_best_ent_loss.pth"
#camvid_weight="/tmp/runs/domain_gap/camvid/20220725-034846/deeplabv3_mobilenet_v3_large_camvid_best_ent_loss.pth"
#cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220801-233634/deeplabv3_mobilenet_v3_large_cityscapes_best_ent_loss.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220728-160250/deeplabv3_mobilenet_v3_large_forest_best_ent_loss.pth"

#camvid_model="deeplabv3_resnet50"
#cityscapes_model="deeplabv3_resnet50"
#forest_model="deeplabv3_resnet50"
#camvid_model="espnetv2"
#cityscapes_model="espnetv2"
#forest_model="espnetv2"
model="espnetv2"
camvid_model=${model}
cityscapes_model=${model}
forest_model=${model}

model_type="best_ent_loss"

# DeepLab-MobileNet
if [ ${model} = "unet" ]; then
    # UNet
    camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221010-201507/${camvid_model}_camvid_${model_type}.pth"
    #camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221001-213201/${camvid_model}_camvid_best_ent_loss.pth"
    cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221006-232113/${cityscapes_model}_cityscapes_${model_type}.pth"
    forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221011-180741/${forest_model}_forest_${model_type}.pth"
elif [ ${model} = "espnetv2" ]; then
    # ESPNetv2
    # camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20221001-213201/${camvid_model}_camvid_${model_type}.pth"
    # cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20221005-203157/${cityscapes_model}_cityscapes_${model_type}.pth"
    # forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20221004-104824/${forest_model}_forest_${model_type}.pth"
    camvid_weight="./pretrained_weights/espdnetue_2.0_480_best_camvid.pth"
    cityscapes_weight="./pretrained_weights/espdnetue_2.0_512_best_city.pth"
    forest_weight="./pretrained_weights/espdnetue_2.0_480_best_forest.pth"
elif [ ${model} = "deeplabv3_mobilenet_v3_large" ]; then
    camvid_weight="/tmp/runs/domain_gap/camvid/${camvid_model}/20220725-034846/${camvid_model}_camvid_best_iou.pth"
    cityscapes_weight="/tmp/runs/domain_gap/cityscapes/${cityscapes_model}/20220801-233634/${cityscapes_model}_cityscapes_best_iou.pth"
    forest_weight="/tmp/runs/domain_gap/forest/${forest_model}/20220728-160250/${forest_model}_forest_best_iou.pth"
fi

python evaluate_domain_gap.py \
    --device cuda \
    --target target \
    --target-data-list ./dataset/data_list/val_greenhouse_a.lst \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --save-path ./pseudo_labels/
