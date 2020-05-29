import shutil
import os
import cv2 as cv
import xmltodict
import numpy as np
import skimage.draw
from pathlib import Path
from utils import poly_from_str
import collections

DATA_DIR = 'data/raw'
OUT_DIR = 'data/prepared_png'


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
                if not isinstance(poly, collections.OrderedDict):
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
    cvat_to_mask('data/data_cvat_only_vesicles/annotations/annotations.xml')
