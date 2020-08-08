# Mask R-CNN Vesicle Segmentation
Mask R-CNN based extracellular vesicle instance segmentation
![Sample](assets/sample_vis.jpg)

The repository includes:
* [detector.py](detector.py) Code for train and evaluation Mask R-CNN based on https://github.com/matterport/Mask_RCNN
* [server.py](server.py) Simple web interface based on Flask. You can see hosted application [here](https://www.bioeng.ru/exosomes/)
* [vesicle.py](vesicle.py) Command line tool for segmentation
* [Dataset](https://github.com/High-resolution-microscopy-laboratory/exosomes/releases/download/v1.0/dataset.zip)
* [Trained model](https://github.com/High-resolution-microscopy-laboratory/exosomes/releases/download/v1.0/mask_rcnn_vesicle_0026.h5)
