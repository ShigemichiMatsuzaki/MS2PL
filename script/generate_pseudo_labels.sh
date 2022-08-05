#camvid_weight="/tmp/runs/domain_gap/camvid/20220614-094914/camvid_ent_best.pth"
camvid_weight="/tmp/runs/domain_gap/camvid/20220610-121207/camvid_ent_best.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220611-115709/cityscapes_ent_best.pth"
forest_weight="/tmp/runs/domain_gap/forest/20220612-133456/forest_ent_best.pth"
camvid_weight="/tmp/runs/domain_gap/camvid/20220630-113848/camvid_ent_current.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220630-222203/cityscapes_ent_current.pth"
forest_weight="/tmp/runs/domain_gap/forest/20220701-231435/forest_ent_current.pth"
camvid_weight="/tmp/runs/domain_gap/camvid/20220725-034846/deeplabv3_mobilenet_v3_large_camvid_best_iou.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220723-154300/deeplabv3_mobilenet_v3_large_cityscapes_best_iou.pth"
# forest_weight="/tmp/runs/domain_gap/forest/20220722-155508/deeplabv3_mobilenet_v3_large_forest_best_ent_loss.pth"
forest_weight="/tmp/runs/domain_gap/forest/20220728-160250/deeplabv3_mobilenet_v3_large_forest_best_iou.pth"

# camvid_weight="/tmp/runs/domain_gap/camvid/20220725-034846/deeplabv3_mobilenet_v3_large_camvid_best_iou.pth"
# cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220723-154300/deeplabv3_mobilenet_v3_large_cityscapes_best_iou.pth"
# forest_weight="/tmp/runs/domain_gap/forest/20220728-160250/deeplabv3_mobilenet_v3_large_forest_best_iou.pth"
camvid_weight="/tmp/runs/domain_gap/camvid/20220801-042638/deeplabv3_mobilenet_v3_large_camvid_best_iou.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220801-233634/deeplabv3_mobilenet_v3_large_cityscapes_best_iou.pth"
forest_weight="/tmp/runs/domain_gap/forest/20220728-160250/deeplabv3_mobilenet_v3_large_forest_best_iou.pth"

camvid_model="deeplabv3_resnet50"
cityscapes_model="deeplabv3_resnet50"
forest_model="deeplabv3_resnet50"
camvid_model="deeplabv3_mobilenet_v3_large"
cityscapes_model="deeplabv3_mobilenet_v3_large"
forest_model="deeplabv3_mobilenet_v3_large"

python generate_pseudo_labels.py \
    --device cuda \
    --target-data-list ./dataset/data_list/train_greenhouse_a.lst \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --batch-size 64 \
    --save-path ./pseudo_labels/mobilenet/