FROM nvcr.io/nvidia/pytorch:22.11-py3

# Add path of edgenets to PYTHONPATH so that the models can be successfully imported
# I believe there should be a better way
ENV PYTHONPATH /root/training/models/edgenets:${PYTHONPATH}

RUN pip3 install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu11.4/torch1.10/index.html

# MMCV: Library for computer vision including MMSegmentation
RUN pip3 uninstall -y pillow
RUN pip3 install -U \
    tqdm \
    torchsummary \
    umap-learn \
    llvmlite \
    pillow-simd \
    openmim \
    mmsegmentation \
    timm \
    optuna \
    albumentations \
    pytorch-lightning \
    fvcore \
    torch-tb-profiler \
    --ignore-installed