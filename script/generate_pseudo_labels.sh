#camvid_weight="/tmp/runs/domain_gap/camvid/20220614-094914/camvid_ent_best.pth"
camvid_weight="/tmp/runs/domain_gap/camvid/20220610-121207/camvid_ent_best.pth"
cityscapes_weight="/tmp/runs/domain_gap/cityscapes/20220611-115709/cityscapes_ent_best.pth"
forest_weight="/tmp/runs/domain_gap/forest/20220612-133456/forest_ent_best.pth"
camvid_model="deeplabv3_resnet50"
cityscapes_model="deeplabv3_resnet50"
forest_model="deeplabv3_resnet50"

python generate_pseudo_labels.py \
    --device cuda \
    --target-data-list ./dataset/data_list/train_greenhouse_a.lst \
    --source-model-names ${camvid_model},${cityscapes_model},${forest_model} \
    --source-dataset-names camvid,cityscapes,forest \
    --source-weight-names ${camvid_weight},${cityscapes_weight},${forest_weight} \
    --batch-size 48 \
    --save-path ./pseudo_labels/