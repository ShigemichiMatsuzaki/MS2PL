FROM nvcr.io/nvidia/pytorch:21.09-py3

# MMCV: Library for computer vision including MMSegmentation
RUN pip3 install -U openmim
RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu11.4/torch1.10/index.html
# RUN mim install mmcv-full
RUN pip3 install mmsegmentation

# Training utilities
RUN pip3 install timm optuna albumentations pytorch-lightning
RUN pip3 install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

# Add path of edgenets to PYTHONPATH so that the models can be successfully imported
# I believe there should be a better way
ENV PYTHONPATH /root/training/models/edgenets:${PYTHONPATH}