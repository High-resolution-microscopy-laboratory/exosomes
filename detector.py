"""
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 detector.py train --dataset=/path/to/vesicles/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 detector.py train --dataset=/path/to/vesicles/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 detector.py train --dataset=/path/to/vesicles/dataset --weights=imagenet

"""
import os
import sys
from collections import OrderedDict

import keras
import tensorflow as tf
import numpy as np
import skimage.draw
import skimage.io
import xmltodict
from imgaug import augmenters as iaa

from utils import poly_from_str, roundness, get_contours

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


class VesicleGrayConfig(VesicleConfig):
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([148.4])


class VesicleInferenceConfig(VesicleConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "none"


############################################################
#  Dataset
############################################################

class VesicleDataset(utils.Dataset):
    # def load_image(self, image_id):
    #     return cv.imread(self.image_info[image_id]['path'], 0)[..., np.newaxis]

    def load_vesicle(self, dataset_dir, subset):
        """Load a subset of the Vesicles dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("vesicle", 1, "vesicle")

        # Train or validation dataset?
        assert subset in ["train", "val"]
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


class MeanAveragePrecisionCallback(keras.callbacks.Callback):
    def __init__(self, train_model: modellib.MaskRCNN, inference_model: modellib.MaskRCNN, dataset: utils.Dataset,
                 calculate_map_at_every_X_epoch=5, dataset_limit=None,
                 verbose=1):
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.calculate_map_at_every_X_epoch = calculate_map_at_every_X_epoch
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the bacth size of 1")

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def on_epoch_end(self, epoch, logs=None):

        if epoch > 2 and (epoch + 1) % self.calculate_map_at_every_X_epoch == 0:
            self._verbose_print("Calculating mAP...")
            self._load_weights_for_model()

            mAPs = self._calculate_mean_average_precision()
            mAP = np.mean(mAPs)

            if logs is not None:
                logs["val_mean_average_precision"] = mAP

            self._verbose_print("mAP at epoch {0} is: {1}".format(epoch + 1, mAP))

        super().on_epoch_end(epoch, logs)

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path,
                                          by_name=True)

    def _calculate_mean_average_precision(self):
        mAPs = []

        # Use a random subset of the data when a limit is defined
        np.random.shuffle(self.dataset_image_ids)

        for image_id in self.dataset_image_ids[:self.dataset_limit]:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.dataset,
                                                                                      self.inference_model.config,
                                                                                      image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, self.inference_model.config), 0)
            results = self.inference_model.detect(molded_images, verbose=0)
            r = results[0]
            # Compute mAP - VOC uses IoU 0.5
            AP, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                           r["class_ids"], r["scores"], r['masks'])
            mAPs.append(AP)

        return np.array(mAPs)


class BoardCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, train_model: modellib.MaskRCNN, inference_model: modellib.MaskRCNN, dataset: utils.Dataset,
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
        self.writer = tf.summary.FileWriter(log_dir)

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the bacth size of 1")

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path,
                                          by_name=True)

    def on_epoch_end(self, epoch, logs=None):
        # super().on_epoch_end(epoch, logs)
        self._verbose_print("Calculating metrics...")
        self._load_weights_for_model()

        metrics = compute_metrics(self.inference_model, self.dataset)
        summary = tf.Summary()
        for key, value in metrics.items():
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = key
        self.writer.add_summary(summary, epoch)
        self.writer.flush()


def train(model, epochs=EPOCHS):
    """Train the model."""

    # Training dataset.
    dataset_train = VesicleDataset()
    dataset_train.load_vesicle(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VesicleDataset()
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

    mean_average_precision_callback = MeanAveragePrecisionCallback(model,
                                                                   inference_model, dataset_val,
                                                                   calculate_map_at_every_X_epoch=1,
                                                                   verbose=1)

    board_callback = BoardCallback(model.log_dir + '_metrics', model, inference_model, dataset_val)

    print("Training network")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                augmentation=augmentation,
                custom_callbacks=[board_callback],
                layers='heads')


# noinspection PyPep8Naming
def compute_metrics(model: modellib.MaskRCNN, dataset: modellib.utils.Dataset, limit=0):
    mAPs = []
    APs_75 = []
    APs_5 = []
    recalls_75 = []
    recalls_5 = []
    roundness_list = []

    end = limit if limit != 0 else len(dataset.image_ids)
    for image_id in dataset.image_ids[:end]:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)

        # Compute metrics
        r = results[0]

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

    return {name: np.mean(values) for name in
            ['mAP@IoU Rng', 'mAP@IoU=75', 'mAP@IoU=50', 'Recall @ IoU=75', 'Recall @ IoU=50', 'Roundness']
            for values in [mAPs, APs_75, APs_5, recalls_75, recalls_5, roundness_list]}


def f_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def avg_roundness(contours) -> float:
    if len(contours) > 0:
        return sum([roundness(cnt) for cnt in contours]) / len(contours)
    else:
        return 0


# noinspection PyPep8Naming
def evaluate(dataset: VesicleDataset, tag='', limit=0, out=None):
    mAP, mAP_75, mAP_5, recall_75, recall_5, roundness = compute_metrics(model, dataset, limit)
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
    parser.add_argument('--out', required=False, type=str)
    parser.add_argument('--tag', required=False, type=str)

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Epochs: ", args.epochs)

    # Configurations
    if args.command == "train":
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
    if args.command == "train":
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
            # 'conv1',
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.epochs)
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = VesicleDataset()
        dataset_val.load_vesicle(args.dataset, 'val')
        dataset_val.prepare()
        n_img = len(dataset_val.image_ids) if not args.eval_limit else len(dataset_val.image_ids[:args.eval_limit])
        print(f'Running evaluation on {n_img} images.')
        evaluate(dataset_val, tag=args.tag, limit=args.eval_limit, out=args.out)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
