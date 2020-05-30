import shutil
import os
import cv2 as cv
import xmltodict
import numpy as np
import skimage.draw
from pathlib import Path
from utils import poly_from_str
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import shutil
from matplotlib import pyplot as plt

DATA_DIR = 'data/raw'
OUT_DIR = 'data/prepared_png'


def create_empty_ann(source_ann):
    empty_ann = OrderedDict()
    empty_ann['annotations'] = OrderedDict()
    root = empty_ann['annotations']
    root['version'] = source_ann['version']
    root['meta'] = source_ann['meta']
    root['image'] = []
    return empty_ann


def split_data(img_dir, xml_path, output_dir):
    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and '.tif' in f]
    scales = [int(s.lower().rsplit('k', 1)[0].rsplit('_', 1)[1]) for s in files]
    train_files, test_files, train_scales, test_scales = train_test_split(files, scales, test_size=0.15,
                                                                          stratify=scales)
    with open(xml_path) as f:
        doc = xmltodict.parse(f.read())
        annotations = doc['annotations']

    # Prepare empty annotation
    train_ann = create_empty_ann(annotations)
    test_ann = create_empty_ann(annotations)

    train_img_dir = os.path.join(output_dir, 'train')
    test_img_dir = os.path.join(output_dir, 'val')

    for image in annotations['image']:
        name = image['@name']
        img_path = os.path.join(img_dir, name)
        if name in train_files:
            train_ann['annotations']['image'].append(image)
            shutil.copy(img_path, train_img_dir)
        if name in test_files:
            test_ann['annotations']['image'].append(image)
            shutil.copy(img_path, test_img_dir)

    train_ann_path = os.path.join(output_dir, 'train', 'annotations.xml')
    test_ann_path = os.path.join(output_dir, 'val', 'annotations.xml')

    with open(train_ann_path, 'w') as f:
        f.write(xmltodict.unparse(train_ann))
    with open(test_ann_path, 'w') as f:
        f.write(xmltodict.unparse(test_ann))

    print(f'train: {len(train_files)}')
    print(f'test: {len(test_files)}')

    plt.hist(scales)
    plt.hist(train_scales)
    plt.hist(test_scales)
    plt.show()


def cvat_to_mask(xml_path):
    cv.namedWindow("mask", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("mask", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    with open(xml_path) as f:
        doc = xmltodict.parse(f.read())
        annotations = doc['annotations']
        for image in annotations['image']:
            name = image['@name']
            if 'polygon' not in image:
                continue
            polygons = image['polygon']
            h = int(image['@height'])
            w = int(image['@width'])
            masks = np.zeros([h, w, len(polygons)], dtype=np.uint8)
            for i, poly in enumerate(polygons):
                if not isinstance(poly, OrderedDict):
                    continue
                mask = np.zeros([h, w, 1], dtype=np.uint8)
                str_points = poly['@points']
                xs, ys = poly_from_str(str_points)
                rr, cc = skimage.draw.polygon(ys, xs)
                mask[rr, cc, 0] = 255
                cv.imshow('mask', mask)
                cv.waitKey(0)


def main():
    dirs = [OUT_DIR]
    for d in dirs:
        if Path(d).exists():
            shutil.rmtree(d)
        Path(d).mkdir()

    img_paths = []
    files = []
    scales = []
    for subdir in os.listdir(DATA_DIR):
        if os.path.isdir(os.path.join(DATA_DIR, subdir)):
            for filename in os.listdir(os.path.join(DATA_DIR, subdir)):
                filepath = os.path.join(DATA_DIR, subdir, filename)
                if os.path.isfile(filepath):
                    img = cv.imread(filepath)
                    new_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                    files.append(filename)
                    # new_filename = filename.replace('tif', 'png')
                    new_filename = filename
                    cv.imwrite(os.path.join(OUT_DIR, f'{subdir}_{new_filename}'), img)

    scales = [f.upper().split('.')[1][:-1] for f in files]
    print(len(files))


if __name__ == '__main__':
    # cvat_to_mask('data/data_cvat_only_vesicles/annotations/annotations.xml')
    split_data('data/data_cvat_only_vesicles/all',
               'data/data_cvat_only_vesicles/all/annotations.xml', 'data/data_cvat_only_vesicles/')
