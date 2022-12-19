# Multi-Source Soft Pseudo-Label Learning

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