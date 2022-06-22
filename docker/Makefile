NAME=domain-gap-eval
IMAGE_NAME=nvcr.io/nvidia/pytorch
CONTAINER_NAME=domain-gap-eval
VERSION=21.09-py3
GPU_ID=1
DATASET_ROOT=/data/aisl/matsuzaki/dataset
RUNS_ROOT=/data/aisl/matsuzaki/runs

build:
	docker build -t $(IMAGE_NAME):$(VERSION) .

train-camvid:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 10G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-camvid \
		$(IMAGE_NAME):$(VERSION) \
		bash script/train_camvid.sh

train-cityscapes:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-cityscapes \
		$(IMAGE_NAME):$(VERSION) \
		bash script/train_cityscapes.sh

train-forest:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 10G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-forest \
		$(IMAGE_NAME):$(VERSION) \
		bash script/train_forest.sh

generate-pseudo-labels:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 10G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-generate-pseudo-labels \
		$(IMAGE_NAME):$(VERSION) \
		bash script/generate_pseudo_labels.sh
