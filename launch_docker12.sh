docker run --gpus '"device=0, 1"'  -u $(id -u):$(id -g) \
				 --shm-size 32G \
				 --log-driver=none \
				 --rm -v /path/to/data:/workspace/data \
				 -v /path/to/src/:/workspace/rlh \
				 -it rlh
