MODEL=espnetv2
SOURCE_MODEL=espnetv2

# Parameters
label_weight_temp=5 
label_weight_threshold=0.1

for use_kld_class_loss in true false; do
for use_label_ent_weight in true false; do

if [ ${use_kld_class_loss} = true ] && [ ${use_label_ent_weight} = true ]; then
    continue
fi
if [ ${use_kld_class_loss} = false ] && [ ${use_label_ent_weight} = false ]; then
    is_hard=true
else
    is_hard=false
fi

if [ ${use_kld_class_loss} = true ]; then
    conf_thresh=0.80
else
    conf_thresh=0.95
fi
python train_pseudo.py \
    --device cuda \
    --model ${MODEL} \
    --use-cosine true \
    --target greenhouse \
    --batch-size 64 \
    --epoch 50 \
    --lr 0.009 \
    --label-update-epoch 5 \
    --save-path /tmp/runs/domain_gap/ \
    --scheduler constant \
    --class-wts-type uniform \
    --is-hard ${is_hard} \
    --use-kld-class-loss ${use_kld_class_loss} \
    --use-label-ent-weight ${use_label_ent_weight} \
    --conf-thresh ${conf_thresh} \
    --use-prototype-denoising true \
    --label-weight-temperature ${label_weight_temp} \
    --label-weight-threshold ${label_weight_threshold} \
    --kld-loss-weight 0.2 \
    --entropy-loss-weight 0.5 \
    --sp-label-min-portion 0.9 \
    --pseudo-label-dir ./pseudo_labels/${SOURCE_MODEL}/ \
    --ignore-index 3 
done
done