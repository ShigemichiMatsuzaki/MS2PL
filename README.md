# Multi-Source Soft Pseudo-Label Learning with Domain Similarity-based Weighting for Semantic Segmentation

## Overview
Domain adaptive training for semantic segmentation network using multiple source datasets.

## Requirements

- NVIDIA GPU with at least xx GB memory
- nvidia-docker

## Usage
### Building an image

Dockerfile provided in this repo uses `nvcr.io/nvidia/pytorch` image as a base image.
```
make build # Built with nvcr.io/nvidia/pytorch:22.11-py3
```
Optionally, you can specify the version of base image.
```
make build VERSION=21.09-py3 # Built with nvcr.io/nvidia/pytorch:21.09-py3
```
**Note**: There could be some compatibility issues when using different versions than `22.11-py3`.
I prepared Dockerimage for building an image from `20.03-py3` with limited libraries (`Dockerfile_2003`). To build it, run the following command:
```
make build-2003
```

### Training a model

#### Pre-training

Train a model for each source dataset using an ordinary supervised learning.
```
make train-<source name> \ # source name: camvid, cityscapes, or forest
    VERSION=<version>
```

#### Pseudo-label generation
```
make generate-pseudo-labels VERSION=<version>
```
Pseudo-labels will be generated under `pseudo_labels` directory.
To change parameters, modify `scripts/generate_pseudo_labels.sh`.

#### Target model training

```
make train-greenhouse-pseudo-soft # Using soft pseudo-labels
make train-greenhouse-pseudo-hard # Using hard pseudo-labels
```
To change parameters, modify `scripts/train_greenhouse_soft_pseudo.sh` / `scripts/train_greenhouse_hard_pseudo.sh`.

## Publication

```
@report{
  author  = {Matsuzaki, Shigemichi and Masuzawa, Hiroaki and Miura, Jun},
  arxivid = {2303.00979},
  title   = {{Multi-Source Soft Pseudo-Label Learning with Domain Similarity-based Weighting for Semantic Segmentation}},
  url     = {http://arxiv.org/abs/2303.00979},
  year    = {2023}
}
```