# Mask R-CNN Vesicle Segmentation
Mask R-CNN based extracellular vesicle instance segmentation
![Sample](assets/sample_vis.jpg)

The repository includes:
* [detector.py](detector.py) Code for train and evaluation Mask R-CNN based on https://github.com/matterport/Mask_RCNN
* [server.py](server.py) Simple web interface based on Flask. You can see hosted application [here](https://www.bioeng.ru/exosomes/)
* [vesicle.py](vesicle.py) Command line tool for segmentation
* [Dataset](https://github.com/High-resolution-microscopy-laboratory/exosomes/releases/download/v1.0/dataset.zip)
* [Trained model](https://github.com/High-resolution-microscopy-laboratory/exosomes/releases/download/v1.0/mask_rcnn_vesicle.h5)


## Run web app in docker

1. Install nvidia docker https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-docker-ce
2. Download [model weights](https://github.com/High-resolution-microscopy-laboratory/exosomes/releases/download/v1.0/mask_rcnn_vesicle.h5)

Run on cpu
```shell script
# Change to actual model absolute path
MODEL_PATH="/path/to/mask_rcnn_vesicle.h5"
docker run \
-v ${MODEL_PATH}:/app/models/mask_rcnn_vesicle.h5 \
-p 8000:8000 \
highresolutionimaging/vesicles
```

Run on gpu
```shell script
# Change to actual model absolute path
MODEL_PATH="/path/to/mask_rcnn_vesicle.h5"
docker run \
-v ${MODEL_PATH}:/app/models/mask_rcnn_vesicle.h5 \
-p 8000:8000 \
--gpus all \
--env TF_FORCE_GPU_ALLOW_GROWTH=true \
highresolutionimaging/vesicles
```

Server listening on 0.0.0.0:8000 so you can access app on localhost:8000 or {HOST_IP}:8000

