#!/usr/bin/env python3

import os
import shutil

import mrcnn.model as modellib
import tensorflow as tf

import detector
import utils
from result import ResultWrapper
import time

DEFAULT_FILE_NAME = 'via_region_data.json'
DETECTOR_FILE_NAME = 'via_region_data_detect.json'
DEFAULT_MODEL_PATH = 'models/final2.h5'


def load_model(weights_path) -> modellib.MaskRCNN:
    # Загрузка модели
    config = detector.VesicleInferenceConfig()
    with tf.device('/cpu:0'):
        model = modellib.MaskRCNN(mode="inference", model_dir='model', config=config)
    model.load_weights(weights_path, by_name=True)

    return model


def detect(images: utils.Images, output_dir, model: modellib.MaskRCNN):
    # Детекция
    results_dict = {}
    for i, name in enumerate(images):
        img = images[name]
        results_dict[name], = model.detect([img], verbose=0)
        print('{}/{}\t{}'.format(i + 1, len(images), name))

    # Экспорт разметки
    os.makedirs(output_dir, exist_ok=True)
    results = ResultWrapper.from_result(results_dict, images)
    results.visualize(images)
    results.save_annotation(os.path.join(output_dir, DETECTOR_FILE_NAME))
    results.save_table(output_dir)
    print("""
        {} saved in
        {}/
        """.format(DETECTOR_FILE_NAME,
                   os.path.abspath(output_dir)))


def detect_from_dir(input_dir, output_dir, model: modellib.MaskRCNN):
    # Подготовка изображений
    prepared_images_dir = output_dir
    os.makedirs(prepared_images_dir, exist_ok=True)
    utils.prepare_images(input_dir, prepared_images_dir)
    images = utils.load_images(prepared_images_dir)
    begin = time.time()
    detect(images, output_dir, model)
    end = time.time()
    n = len(images)
    duration = end - begin
    print(f'Processed {n} in {duration} sec')
    print(f'{duration / n} sec per image')
    utils.write_images(images, os.path.join(output_dir, 'vis'))


def export_results(input_dir, out_dir):
    input_path = os.path.join(input_dir, DEFAULT_FILE_NAME)
    output_path = os.path.join(out_dir, DEFAULT_FILE_NAME)

    if not os.path.exists(input_path):
        input_path = os.path.join(input_dir, DETECTOR_FILE_NAME)
        shutil.copy(input_path, output_path)
    else:
        shutil.move(input_path, output_path)

    results = ResultWrapper.from_json(output_path, [])
    results.save_table(out_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Mask R-CNN to detect vesicle')
    parser.add_argument('command',
                        help="'detect' or 'export'")
    parser.add_argument('input_dir',
                        help='Directory of the input images')
    parser.add_argument('output_dir', nargs='?',
                        help='Directory for the output data')
    parser.add_argument('--weights', required=False,
                        default=DEFAULT_MODEL_PATH,
                        help='Model weights path (default={})'.format(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    if args.command == 'detect':
        if args.input_dir and args.output_dir:
            model = load_model(args.weights)
            detect_from_dir(args.input_dir, args.output_dir, model)

        else:
            print('Arguments input_dir and output_dir are required for detection')

    if args.command == 'export':
        if not args.input_dir:
            print('input_dir is required for export')
        output_dir = args.output_dir or args.input_dir
        export_results(args.input_dir, output_dir)
