NAME=domain-gap-eval
IMAGE_NAME=nvcr.io/nvidia/pytorch
CONTAINER_NAME=domain-gap-eval
# VERSION=21.09-py3
VERSION=22.11-py3
GPU_ID=1
DATASET_ROOT=/data/aisl/matsuzaki/dataset
RUNS_ROOT=/data/aisl/matsuzaki/runs

build:
	docker build -t $(IMAGE_NAME):$(VERSION) .

bash:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 10G \
		--name $(CONTAINER_NAME)-bash \
		$(IMAGE_NAME):$(VERSION) \
		bash 

eval-model:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		--rm \
		--shm-size 10G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-eval-model \
		$(IMAGE_NAME):$(VERSION) \
		bash script/eval_model.sh

train-camvid:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 20G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-camvid \
		$(IMAGE_NAME):$(VERSION) \
		bash script/pretrain.sh camvid

train-cityscapes:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 20G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-cityscapes \
		$(IMAGE_NAME):$(VERSION) \
		bash script/pretrain.sh cityscapes

train-cityscapes-highres:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 20G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-cityscapes-highres \
		$(IMAGE_NAME):$(VERSION) \
		bash script/pretrain_highres.sh cityscapes

train-gta5-highres:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 20G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-gta5-highres \
		$(IMAGE_NAME):$(VERSION) \
		bash script/pretrain_highres.sh gta5

train-forest:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 20G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-forest \
		$(IMAGE_NAME):$(VERSION) \
		bash script/pretrain.sh forest

train-greenhouse-pseudo-hard:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-greenhouse-pseudo-hard \
		$(IMAGE_NAME):$(VERSION) \
		bash script/train_greenhouse_hard_pseudo.sh

train-greenhouse-pseudo-soft:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-greenhouse-pseudo-soft \
		$(IMAGE_NAME):$(VERSION) \
		bash script/train_greenhouse_soft_pseudo.sh

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

evaluate-domain-gap:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 10G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-evaluate-domain-gap \
		$(IMAGE_NAME):$(VERSION) \
		bash script/evaluate_domain_gap.sh

evaluate-source-models:
	docker run -it \
		--gpus="device=all" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 10G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-evaluate-source-models \
		$(IMAGE_NAME):$(VERSION) \
		bash script/evaluate_source_models.sh
