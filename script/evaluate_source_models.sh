#camvid_weight="/tmp/runs/domain_gap/camvid/20220625-120300/camvid_ent_current.pth"
#cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220626-202624/cityscapes_ent_current.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220628-154333/forest_ent_current.pth"
#camvid_weight="/tmp/runs/domain_gap/camvid/20220630-113848/camvid_ent_current.pth"
#cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220630-222203/cityscapes_ent_current.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220701-231435/forest_ent_current.pth"
camvid_weight="/tmp/runs/domain_gap/camvid/20220725-034846/deeplabv3_mobilenet_v3_large_camvid_best_ent_loss.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220723-154300/deeplabv3_mobilenet_v3_large_cityscapes_best_ent_loss.pth"
#forest_weight="/tmp/runs/domain_gap/forest/20220722-155508/deeplabv3_mobilenet_v3_large_forest_best_ent_loss.pth"
forest_weight="/tmp/runs/domain_gap/forest/20220727-220126/deeplabv3_mobilenet_v3_large_forest_best_ent_loss.pth"

# camvid_model="deeplabv3_resnet50"
# cityscapes_model="deeplabv3_resnet50"
# forest_model="deeplabv3_resnet50"
camvid_model="deeplabv3_mobilenet_v3_large"
cityscapes_model="deeplabv3_mobilenet_v3_large"
forest_model="deeplabv3_mobilenet_v3_large"

python evaluate_source_models.py \
    --device cuda \
    --target greenhouse \
    --target-data-list ./dataset/data_list/val_greenhouse_a.lst \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --save-path ./pseudo_labels/