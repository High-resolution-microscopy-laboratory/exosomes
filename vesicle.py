#!/usr/bin/env python3

import utils
import os

# Mask RCNN
import tensorflow as tf
from mrcnn import utils
import mrcnn.model as modellib

import detector
import utils
import result

IMAGES_DIR = 'images'


def detect(input_dir, output_dir, weights_path):
    # Подготовка изображений
    # prepared_images_dir = os.path.join(output_dir, 'images')
    prepared_images_dir = output_dir
    os.makedirs(prepared_images_dir, exist_ok=True)
    utils.prepare_images(input_dir, prepared_images_dir)
    images = utils.load_images(prepared_images_dir)

    # Загрузка модели
    config = detector.VesicleInferenceConfig()
    with tf.device('/cpu:0'):
        model = modellib.MaskRCNN(mode="inference", model_dir='model', config=config)
    model.load_weights(weights_path, by_name=True)

    # Детекция
    results_dict = {}
    for i, name in enumerate(images):
        img = images[name]
        print('{}/{}\t{}'.format(i + 1, len(images), name))
        results_dict[name], = model.detect([img], verbose=0)

    # Экспорт разметки
    results = result.ResultWrapper.from_result(results_dict)
    results.save_annotation(os.path.join(prepared_images_dir, 'via_region_data.json'))
    print('via_region_data.json saved in')
    print(os.path.abspath(prepared_images_dir))


def export_results(out_dir, json_file='via_region_data.json'):
    json_path = os.path.join(out_dir, json_file)
    results = result.ResultWrapper.from_json(json_path)
    table_path = os.path.join(out_dir)
    results.save_table(table_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect vesicle')
    parser.add_argument('command',
                        metavar="'<command>'",
                        help="'detect' or 'export'")
    parser.add_argument('--input_dir', required=True,
                        metavar='/path/to/input/images/',
                        help='Directory of the input images')
    parser.add_argument('--output_dir', required=True,
                        metavar='/path/to/output/dir/',
                        help='Directory for the output data')
    parser.add_argument('--weights', required=False,
                        default='models/final.h5',
                        metavar='/path/to/model/',
                        help='Model weights path (default=models/final.h5)')
    args = parser.parse_args()

    if args.command == 'detect':
        detect(args.input_dir, args.output_dir, args.weights)

    if args.command == 'export':
        export_results(args.output_dir)
