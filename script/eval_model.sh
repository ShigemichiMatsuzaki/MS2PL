# MODEL=deeplabv3_mobilenet_v3_large
MODEL=deeplabv3_resnet50
# MODEL=espnetv2
# MODEL=unet

python eval_model.py \
    --device cuda \
    --model ${MODEL}
