NAME=domain-gap-eval
IMAGE_NAME=nvcr.io/nvidia/pytorch
CONTAINER_NAME=domain-gap-eval
VERSION=20.03-py3

train:
	docker run -it \
		--gpus="device=2" \
		-v ${PWD}/../:/root/training/ \
		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 1G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(VERSION) \
		python train.py --device cuda
