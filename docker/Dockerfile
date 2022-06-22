FROM nvcr.io/nvidia/pytorch:21.09-py3

# MMCV: Library for computer vision including MMSegmentation
RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu11.4/torch1.10/index.html

# Training utilities
RUN pip3 install timm optuna albumentations pytorch-lightning
RUN pip3 install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git