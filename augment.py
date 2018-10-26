import cv2
import os
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, RandomContrast, RandomBrightness, OneOf, Compose
)

import utils


def augmentation(p=1):
    return Compose([
        OneOf([
            RandomBrightness(limit=0.1, p=0.5),
            RandomContrast(limit=0.1, p=0.5),
        ]),
        ShiftScaleRotate(scale_limit=0, border_mode=cv2.BORDER_CONSTANT),
        HorizontalFlip()
    ], p=p)


def apply(img_dir, mask_dir, output_dir, n=5):
    images = utils.load_images(img_dir)
    masks = utils.load_images(mask_dir)
    aug_images = {}
    aug_masks = {}
    for i in range(n):
        for name in images:
            img = images[name]
            mask = masks[name]
            aug = augmentation()
            augmented = aug(image=img, mask=mask)
            aug_images[str(i) + name] = augmented['image']
            aug_masks[str(i) + name] = augmented['mask']

    for name in aug_masks:
        mask = aug_masks[name]
        path = os.path.join(output_dir, name)
        cv2.imwrite(path, mask)

    json_path = os.path.join(output_dir, 'via_region_data.json')
    utils.masks2json(json_path, output_dir)

    for name in aug_masks:
        if '.png' in name:
            path = os.path.join(output_dir, name)
            os.remove(path)

    for name in aug_images:
        img = aug_images[name]
        path = os.path.join(output_dir, name)
        cv2.imwrite(path, img)
