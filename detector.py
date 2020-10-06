#!/usr/bin/env python3

"""
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 detector.py train --dataset=/path/to/vesicles/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 detector.py train --dataset=/path/to/vesicles/dataset --weights=last

"""
import os
import pprint
import sys
from collections import OrderedDict
from typing import Tuple, Iterator

import cv2 as cv
import keras
import numpy as np
import skimage.draw
import skimage.io
import tensorflow as tf
import xmltodict
from imgaug import augmenters as iaa
import neptune
from neptunecontrib.versioning.data import log_data_version
from neptunecontrib.monitoring.keras import NeptuneMonitor

from utils import poly_from_str, roundness, get_contours, visualize_result

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

EPOCHS = 2


############################################################
#  Configurations
############################################################


class VesicleConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "vesicle"

    # We use a GPU with 12GB memory, which can fit two images.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + vesicle

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

    # resnet 50 or resnet101
    BACKBONE = "resnet50"

    LEARNING_RATE = 0.001

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 0.2
    }


class VesicleGrayConfig(VesicleConfig):
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([148.4])


class VesicleInferenceConfig(VesicleConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################

class VesicleDataset(utils.Dataset):
    def load_image(self, image_id):
        img = cv.imread(self.image_info[image_id]['path'], cv.IMREAD_COLOR + cv.IMREAD_ANYDEPTH)
        if img.dtype is np.dtype('uint16'):
            img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        return img

    def load_vesicle(self, dataset_dir, subset):
        """Load a subset of the Vesicles dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("vesicle", 1, "vesicle")

        # Train or validation dataset?
        assert subset in ['train', 'val', 'test']
        dataset_dir = os.path.join(dataset_dir, subset)
        annotation_path = os.path.join(dataset_dir, 'annotations.xml')

        # Load annotations
        with open(annotation_path) as f:
            doc = xmltodict.parse(f.read())
            annotations = doc['annotations']

        # Add images
        for image in annotations['image']:
            name = image['@name']

            if 'polygon' not in image:
                continue

            image_path = os.path.join(dataset_dir, name)
            img = skimage.io.imread(image_path)
            height, width = img.shape[:2]

            polygons = [poly_from_str(str_p['@points']) for str_p in image['polygon']
                        if isinstance(str_p, OrderedDict)]

            self.add_image(
                "vesicle",
                image_id=name,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a vesicle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "vesicle":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p[1], p[0])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "vesicle":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class FRUDataset(VesicleDataset):
    def load_vesicle(self, dataset_dir, subset):
        """Load a subset of the Vesicles dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("vesicle", 1, "vesicle")

        # Train or validation dataset?
        assert subset in ['train', 'val', 'test']
        dataset_dir = os.path.join(dataset_dir, subset)
        annotation_path = os.path.join(dataset_dir, 'masks')

        # Add images
        for name in [file for file in os.listdir(dataset_dir) if 'tif' in file]:
            idx, base_name = name.split('_')
            image_path = os.path.join(dataset_dir, name)
            mask_path = os.path.join(annotation_path, f'{idx}_seg_{base_name}')
            img = skimage.io.imread(image_path)
            height, width = img.shape[:2]

            self.add_image(
                "vesicle",
                image_id=name,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask_path=mask_path)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a vesicle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "vesicle":
            return super(self.__class__, self).load_mask(image_id)

        # Convert 16 bit mask to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        bin_mask = get_bin_mask(mask)
        n_instance = bin_mask.shape[-1]
        return bin_mask, np.ones([n_instance], dtype=np.int32)


class BoardCallback(keras.callbacks.Callback):
    def __init__(self, train_model: modellib.MaskRCNN, inference_model: modellib.MaskRCNN,
                 dataset: utils.Dataset,
                 dataset_limit=None,
                 verbose=1):
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the bacth size of 1")

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path,
                                          by_name=True)


class TensorBoardCallback(BoardCallback):
    def __init__(self, log_dir, train_model: modellib.MaskRCNN, inference_model: modellib.MaskRCNN,
                 dataset: utils.Dataset,
                 dataset_limit=None,
                 verbose=1):

        super().__init__(train_model=train_model,
                         inference_model=inference_model,
                         dataset=dataset,
                         dataset_limit=dataset_limit,
                         verbose=verbose)

        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        self._verbose_print("Calculating metrics...")
        self._load_weights_for_model()

        images, gt_boxes, gt_class_ids, gt_masks, results = detect(self.inference_model, self.dataset)
        metrics = compute_metrics(images, gt_boxes, gt_class_ids, gt_masks, results)

        pprint.pprint(metrics)

        # Images
        summary_img = tf.Summary()
        for i, img in enumerate(images):
            if img.shape[2] != 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            visualize_result(img, results[i])
            _, buf = cv.imencode('.png', img)
            im_summary = tf.Summary.Image(encoded_image_string=buf.tobytes())
            summary_img.value.add(tag=f'img/{i}', image=im_summary)

        # Metrics
        summary = tf.Summary()
        for key, value in metrics:
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = 'Metrics/' + key
        self.writer.add_summary(summary, epoch)
        self.writer.add_summary(summary_img, epoch)
        self.writer.flush()


class NeptuneCallback(BoardCallback):
    def __init__(self, project_name,
                 params,
                 train_model: modellib.MaskRCNN,
                 inference_model: modellib.MaskRCNN,
                 dataset: utils.Dataset,
                 dataset_limit=None,
                 verbose=1):
        super().__init__(train_model=train_model,
                         inference_model=inference_model,
                         dataset=dataset,
                         dataset_limit=dataset_limit,
                         verbose=verbose)

        neptune.init(project_name)
        neptune.create_experiment(project_name, params=params, upload_source_files=['detector.py', 'utils.py'])
        log_data_version(args.dataset)

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path,
                                          by_name=True)

    def on_epoch_end(self, epoch, logs=None):
        self._verbose_print("Calculating metrics...")
        self._load_weights_for_model()

        images, gt_boxes, gt_class_ids, gt_masks, results = detect(self.inference_model, self.dataset)
        metrics = compute_metrics(images, gt_boxes, gt_class_ids, gt_masks, results)

        pprint.pprint(metrics)

        # Images
        for i, img in enumerate(images):
            if img.shape[2] != 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            visualize_result(img, results[i])
            # _, buf = cv.imencode('.png', img)
            cv.imwrite(f'test_{i}.jpg', img)
            neptune.log_image(f'image_epoch_{epoch}', img[..., ::-1])

        # Metrics
        for key, value in metrics:
            neptune.log_metric(key, epoch, value)


def train(model, epochs=EPOCHS, dataset=VesicleDataset):
    """Train the model."""

    # Training dataset.
    dataset_train = dataset()
    dataset_train.load_vesicle(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = dataset()
    dataset_val.load_vesicle(args.dataset, "val")
    dataset_val.prepare()

    augmentation = iaa.SomeOf((0, None), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-180, 180)),
        iaa.Affine(scale=(0.9, 1.1)),
    ])

    inference_model = modellib.MaskRCNN(mode="inference", config=config,
                                        model_dir=args.logs)
    params = args.__dict__
    params.update(config.__dict__)
    neptune_callback = NeptuneCallback("neptunus/exosomes", params, model, inference_model, dataset_val)
    tb_callback = TensorBoardCallback(model.log_dir, model, inference_model, dataset_val)

    custom_callbacks = [tb_callback]
    if args.neptune:
        custom_callbacks += [neptune_callback, NeptuneMonitor()]

    print(f"Training {args.layers} network layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                augmentation=augmentation,
                custom_callbacks=custom_callbacks,
                layers=args.layers)


def train_fru(model, epochs=EPOCHS):
    """Train model on the FRU-Net data."""
    train(model, epochs=epochs, dataset=FRUDataset)


def detect(model: modellib.MaskRCNN, dataset: modellib.utils.Dataset, limit=0):
    gt_boxes = []
    gt_class_ids = []
    gt_masks = []
    results = []
    images = []
    end = limit if limit != 0 else len(dataset.image_ids)
    for image_id in dataset.image_ids[:end]:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        images.append(image)
        gt_boxes.append(gt_bbox)
        gt_class_ids.append(gt_class_id)
        gt_masks.append(gt_mask)

        result = model.detect([image], verbose=0)
        results.append(result[0])

    return images, gt_boxes, gt_class_ids, gt_masks, results


def get_bin_mask(mask):
    n = np.max(mask)
    bin_mask = np.zeros((*mask.shape[:2], n), dtype=np.bool)
    for i in range(1, np.max(mask)):
        bin_mask[:, :, i] = mask == i
    return bin_mask


def draw_mask_cnts(img, masks, color):
    n = masks.shape[2]
    for i in range(n):
        mask = masks[:, :, i]
        _, contours, _ = cv.findContours(mask.copy().astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        cv.drawContours(img, contours, -1, color, 1, cv.LINE_AA)


def visualise(img, mask_gt, mask_det):
    draw_mask_cnts(img, mask_gt, (0, 255, 0))
    draw_mask_cnts(img, mask_det, (0, 0, 255))


def get_fru_net_results(results_dir: str, dataset: VesicleDataset):
    WIN_NAME = 'img'
    cv.namedWindow(WIN_NAME)
    gt_boxes = []
    gt_class_ids = []
    gt_masks = []
    results = []
    images = []

    for image_id in dataset.image_ids:
        origin_img = dataset.load_image(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        gt_boxes.append(gt_bbox)
        gt_class_ids.append(gt_class_id)
        gt_masks.append(gt_mask)

        # load masks
        img_name = dataset.image_info[image_id]['id']
        name, ext = img_name.rsplit('.', 1)
        path = os.path.join(results_dir, f'{name}_labels.{ext}')
        mask_img = cv.imread(path, cv.IMREAD_GRAYSCALE + cv.IMREAD_ANYDEPTH)
        n = np.max(mask_img)
        class_ids = np.ones(n, dtype=np.int32)
        scores = np.ones(n, dtype=np.float32)
        mask = get_bin_mask(mask_img)

        image, window, scale, padding, crop = utils.resize_image(
            origin_img,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        mask = utils.resize_mask(mask, scale, padding, crop)
        rois = utils.extract_bboxes(mask)
        images.append(image)

        vis_img = image.copy()
        visualise(vis_img, gt_mask, mask)
        cv.imshow(WIN_NAME, vis_img)
        cv.waitKey(0)

        result = {
            'class_ids': class_ids,
            'scores': scores,
            'masks': mask,
            'rois': rois
        }
        results.append(result)

    return images, gt_boxes, gt_class_ids, gt_masks, results


# noinspection PyPep8Naming
def compute_metrics(images, gt_boxes, gt_class_ids, gt_masks, results_list) -> Iterator[Tuple]:
    mAPs = []
    APs_75 = []
    APs_5 = []
    recalls_75 = []
    recalls_5 = []
    roundness_list = []

    for image, gt_bbox, gt_class_id, gt_mask, results in zip(images, gt_boxes, gt_class_ids, gt_masks, results_list):
        # Compute metrics
        r = results

        AP_75, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.75)

        AP_5, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.5)

        mAP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                                     r['rois'], r['class_ids'], r['scores'], r['masks'], verbose=False)

        recall_75, _ = utils.compute_recall(r['rois'], gt_bbox, iou=0.75)
        recall_5, _ = utils.compute_recall(r['rois'], gt_bbox, iou=0.5)
        # Roundness
        contours = get_contours(r)
        img_roundness = avg_roundness(contours)

        mAPs.append(mAP)
        APs_75.append(AP_75)
        APs_5.append(AP_5)
        recalls_75.append(recall_75)
        recalls_5.append(recall_5)
        roundness_list.append(img_roundness)

    names = ['1. mAP@IoU Rng', '2. mAP@IoU=75', '3. mAP@IoU=50',
             '4. Recall @ IoU=75', '5. Recall @ IoU=50', '6. Roundness']
    values_list = [mAPs, APs_75, APs_5, recalls_75, recalls_5, roundness_list]
    avg_values = [np.mean(values) for values in values_list]
    print(list(zip(names, avg_values)))

    return zip(names, avg_values)


def f_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def avg_roundness(contours) -> float:
    if len(contours) > 0:
        return sum([roundness(cnt) for cnt in contours]) / len(contours)
    else:
        return 0


# noinspection PyPep8Naming
def evaluate(dataset: VesicleDataset, tag='', limit=0, out=None):
    images, gt_boxes, gt_class_ids, gt_masks, results = detect(model, dataset, limit)
    metrics = compute_metrics(images, gt_boxes, gt_class_ids, gt_masks, results)
    mAP, mAP_75, mAP_5, recall_75, recall_5, roundness = [m[1] for m in metrics]
    F1 = f_score(mAP_75, recall_75)
    print(f'   mAP @ IoU Rng :\t{mAP:.3}')
    print(f'    mAP @ IoU=75 :\t{mAP_75:.3}')
    print(f'    mAP @ IoU=50 :\t{mAP_5:.3}')
    print(f' Recall @ IoU=75 :\t{recall_75:.3}')
    print(f' Recall @ IoU=50 :\t{recall_5:.3}')
    print(f'F Score @ IoU=75 :\t{F1:.3}')
    print(f'       Roundness :\t{roundness:.3}')
    if out:
        with open(out, 'a') as f:
            f.write(f'{tag}, {mAP:.3}, {mAP_75:.3}, {mAP_5:.3}, {recall_75:.3}, {recall_5:.3}, {F1:.3}\n')


def evaluate_fru_net(dataset: VesicleDataset, tag='', limit=0, out=None):
    images, gt_boxes, gt_class_ids, gt_masks, results = get_fru_net_results(
        '/home/ruslan/projects/FRU_processing/code/fru_test_no_bars_results', dataset)
    metrics = compute_metrics(images, gt_boxes, gt_class_ids, gt_masks, results)
    mAP, mAP_75, mAP_5, recall_75, recall_5, roundness = [m[1] for m in metrics]
    F1 = f_score(mAP_75, recall_75)
    print(f'   mAP @ IoU Rng :\t{mAP:.3}')
    print(f'    mAP @ IoU=75 :\t{mAP_75:.3}')
    print(f'    mAP @ IoU=50 :\t{mAP_5:.3}')
    print(f' Recall @ IoU=75 :\t{recall_75:.3}')
    print(f' Recall @ IoU=50 :\t{recall_5:.3}')
    print(f'F Score @ IoU=75 :\t{F1:.3}')
    print(f'       Roundness :\t{roundness:.3}')
    if out:
        with open(out, 'a') as f:
            f.write(f'{tag}, {mAP:.3}, {mAP_75:.3}, {mAP_5:.3}, {recall_75:.3}, {recall_5:.3}, {F1:.3}\n')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect vesicles.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/vesicles/dataset/",
                        help='Directory of the Vesicle dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--epochs', required=False,
                        type=int,
                        default=EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--eval-limit', required=False,
                        type=int,
                        default=0,
                        help='Number of evaluation images')
    parser.add_argument('--layers', required=False,
                        type=str,
                        default='all',
                        help="""
    - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up""")
    parser.add_argument('--out', required=False, type=str)
    parser.add_argument('--tag', required=False, type=str)
    parser.add_argument('--neptune', required=False,
                        type=bool,
                        default=False,
                        help='Enable logging to neptune.io')

    args = parser.parse_args()


    def is_train(command):
        return 'train' in command


    # Validate arguments
    if is_train(args.command):
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Epochs: ", args.epochs)

    # Configurations
    if is_train(args.command):
        config = VesicleConfig()
    else:
        class InferenceConfig(VesicleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if is_train(args.command):
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.epochs)

    if args.command == "train_fru":
        train_fru(model, args.epochs)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = VesicleDataset()
        dataset_val.load_vesicle(args.dataset, 'test')
        dataset_val.prepare()
        n_img = len(dataset_val.image_ids) if not args.eval_limit else len(dataset_val.image_ids[:args.eval_limit])
        print(f'Running evaluation on {n_img} images.')
        evaluate(dataset_val, tag=args.tag, limit=args.eval_limit, out=args.out)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
