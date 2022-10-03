MODEL=deeplabv3_mobilenet_v3_large
# MODEL=espnetv2

python eval_model.py \
    --device cuda \
    --model ${MODEL}