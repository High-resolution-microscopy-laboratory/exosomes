from mrcnn.config import Config


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
    STEPS_PER_EPOCH = 138

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
        "mrcnn_mask_loss": 1
    }

    MEAN_PIXEL = np.array([148.4] * 3)

    def get_dict(self):
        """Return dict with configuration values."""
        return {a: getattr(self, a) for a in dir(self)
                if not a.startswith("__") and not callable(getattr(self, a))}


class FRUConfig(VesicleConfig):
    IMAGE_MIN_DIM = 0
    IMAGE_MAX_DIM = 1024
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 63

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 0.2
    }

    MEAN_PIXEL = np.array([129.4] * 3)


class VesicleGrayConfig(VesicleConfig):
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([148.4])


class VesicleInferenceConfig(VesicleConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
