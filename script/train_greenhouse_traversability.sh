RESUME_FROM=/tmp/runs/domain_gap/pseudo_soft/espnetv2/20230111-231518/pseudo_espnetv2_greenhouse_best_iou.pth
OPTIMIZER_NAME=SGD
SCHEDULER_NAME=constant
LEARNING_RATE=0.009
python train_traversability.py \
    --device cuda \
    --target greenhouse \
    --model esptnet \
    --resume-from ${RESUME_FROM} \
    --optim ${OPTIMIZER_NAME} \
    --scheduler ${SCHEDULER_NAME} \
    --lr ${LEARNING_RATE} \
    --epoch 30 \
    --batch-size 24 \
    --use-cosine true \
    --train-data-list-path dataset/data_list/trav_train_greenhouse_b.lst \
    --val-data-list-path dataset/data_list/trav_test_greenhouse_a.lst \
    --test-data-list-path dataset/data_list/trav_test_greenhouse_a.lst \
    --save-path /tmp/runs/domain_gap/traversability/

