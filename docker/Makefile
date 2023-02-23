NAME=domain-gap-eval
IMAGE_NAME=aisl/ms2pl
CONTAINER_NAME=domain-gap-eval
VERSION=22.11-py3
GPU_ID=all
DATASET_ROOT=/media/aisl/ExtremeSSD/dataset
RUNS_ROOT=/data/aisl/matsuzaki/runs

build:
	docker build -t $(IMAGE_NAME):$(VERSION) .

# For older drivers, with limited libraries
build-2003:
	docker build -t $(IMAGE_NAME):20.03-py3 -f Dockerfile_2003 .

bash:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
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
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		--rm \
		--shm-size 10G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-eval-model \
		$(IMAGE_NAME):$(VERSION) \
		bash script/eval_model.sh

train-camvid:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
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
		--gpus '"device=${GPU_ID}"' \
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
		--gpus '"device=${GPU_ID}"' \
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
		--gpus '"device=${GPU_ID}"' \
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
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 20G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-forest \
		$(IMAGE_NAME):$(VERSION) \
		bash script/pretrain.sh forest
	
train-joint:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 20G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-joint \
		$(IMAGE_NAME):$(VERSION) \
		bash script/pretrain_joint.sh 


train-greenhouse-pseudo-hard:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
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
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-greenhouse-pseudo-soft-2 \
		$(IMAGE_NAME):$(VERSION) \
		bash script/train_greenhouse_soft_pseudo.sh

train-sakaki-pseudo-soft:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-sakaki-pseudo-soft \
		$(IMAGE_NAME):$(VERSION) \
		bash script/train_sakaki_soft_pseudo.sh


generate-pseudo-labels:
	docker run -it \
		--gpus="device=${GPU_ID}" \
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
		--gpus="device=${GPU_ID}" \
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
		--gpus="device=${GPU_ID}" \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 10G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-evaluate-source-models \
		$(IMAGE_NAME):$(VERSION) \
		bash script/evaluate_source_models.sh

optuna-dashboard:
	docker run -it \
		--rm \
		-p 8080:8080 \
		-v ${PWD}/../:/app \
		-w /app \
		ghcr.io/optuna/optuna-dashboard:v0.9.0b2 \
		sqlite:///pseudo_soft_espnetv2.db

train-greenhouse-traversability:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-greenhouse-traversability \
		$(IMAGE_NAME):$(VERSION) \
		bash script/train_greenhouse_traversability.sh

msdacl-test:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset/ \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-msdacl-test \
		$(IMAGE_NAME):$(VERSION) \
		bash baselines/msdacl_test.sh

#
# Experiment
#
comparative-park:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset/ \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-comparative-park \
		$(IMAGE_NAME):$(VERSION) \
		bash script/exp/comparative_park.sh

comparative-park-single:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset/ \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-comparative-park-single \
		$(IMAGE_NAME):$(VERSION) \
		bash script/exp/comparative_park_single.sh


comparative-greenhouse:
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/../:/root/training/ \
		-v ${DATASET_ROOT}:/tmp/dataset/ \
		-v ${RUNS_ROOT}:/tmp/runs/ \
		--rm \
		--shm-size 15G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME)-comparative-greenhouse \
		$(IMAGE_NAME):$(VERSION) \
		bash script/exp/comparative_greenhouse.sh
