import seaborn as sns
import os
import xmltodict
import numpy as np
from utils import poly_from_str
import cv2 as cv
import skimage.draw
from collections import OrderedDict
from matplotlib import pyplot as plt
import pandas as pd

DATA_DIR = 'data/dataset'
SUBSETS = ['train', 'val', 'test']
SHOWIMG = False
WIN_NAME = 'mask'
RESIZE = 1024

sns.set_theme()


def load_annotations(data_dir, subset) -> dict:
    path = os.path.join(data_dir, subset, 'annotations.xml')
    with open(path) as f:
        doc = xmltodict.parse(f.read())
        return doc['annotations']


def get_new_size(h, w, max_size=1024):
    ar = w / h
    if h > w:
        h = max_size
        w = h * ar
    else:
        w = max_size
        h = w / ar
    return int(h), int(w)


def get_stats(annotations, showimg=False, resize=1024):
    n_images = 0
    n_instances = 0
    areas = []
    for image in annotations['image']:
        name = image['@name']
        if 'polygon' not in image:
            continue
        polygons = image['polygon']
        h = int(image['@height'])
        w = int(image['@width'])
        if resize:
            new_h, new_w = get_new_size(h, w, resize)
        else:
            new_h, new_w = h, w
        masks = np.zeros([new_h, new_w, 3], dtype=np.uint8)
        n_images += 1
        for i, poly in enumerate(polygons):
            mask = np.zeros([new_h, new_w, 1], dtype=np.uint8)
            if not isinstance(poly, OrderedDict):
                continue
            n_instances += 1
            str_points = poly['@points']
            xs, ys = poly_from_str(str_points)
            if resize:
                xs = xs / w * new_w
                ys = ys / h * new_h
            rr, cc = skimage.draw.polygon(ys, xs)
            mask[rr, cc, 0] = 255
            _, cnt, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            area = len(rr)
            areas.append(area)
            if showimg:
                cv.drawContours(masks, cnt, 0, (0, 255, 0), lineType=cv.LINE_AA)
                rect = cv.boundingRect(cnt[0])
                point1 = tuple([rect[0], rect[1] - 3])
                cv.putText(masks, str(area), point1, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv.LINE_AA)
        if showimg:
            cv.imshow(WIN_NAME, masks)
            cv.setWindowTitle(WIN_NAME, name)
            cv.waitKey(0)
    return n_images, n_instances, areas


for set_name in SUBSETS:
    if SHOWIMG:
        cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
    print(f'{set_name} set')
    n_images, n_instances, areas = get_stats(load_annotations(DATA_DIR, set_name), showimg=SHOWIMG, resize=RESIZE)
    print(f'images: {n_images} instances: {n_instances}')
    areas_df = pd.DataFrame(areas)
    areas_df.to_csv(f'{set_name}_no_resize.csv', header=False, index=False)
    print(areas_df.describe())
    sns.histplot(areas).set_title(f'{set_name} areas')
    plt.show()
