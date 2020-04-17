#!/usr/bin/env python3

import os
import shutil
import result
import utils

DEFAULT_FILE_NAME = 'via_region_data.json'
DETECTOR_FILE_NAME = 'via_region_data_detect.json'
DEFAULT_MODEL_PATH = 'models/final.h5'


def load_model():
    pass


def detect(images: utils.Images, output_dir):
    pass


def detect_from_dir(input_dir, output_dir, weights_path):
    # Mask RCNN
    import tensorflow as tf
    import mrcnn.model as modellib

    import detector

    # Подготовка изображений
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
        results_dict[name], = model.detect([img], verbose=0)
        print('{}/{}\t{}'.format(i + 1, len(images), name))

    # Экспорт разметки
    results = result.ResultWrapper.from_result(results_dict)
    results.save_annotation(os.path.join(prepared_images_dir, DETECTOR_FILE_NAME))
    results.save_table(prepared_images_dir)
    print("""
    {} saved in
    {}/
    """.format(DETECTOR_FILE_NAME,
               os.path.abspath(prepared_images_dir)))


def export_results(input_dir, out_dir):
    input_path = os.path.join(input_dir, DEFAULT_FILE_NAME)
    output_path = os.path.join(out_dir, DEFAULT_FILE_NAME)

    if not os.path.exists(input_path):
        input_path = os.path.join(input_dir, DETECTOR_FILE_NAME)
        shutil.copy(input_path, output_path)
    else:
        shutil.move(input_path, output_path)

    results = result.ResultWrapper.from_json(output_path)
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
            detect_from_dir(args.input_dir, args.output_dir, args.weights)
        else:
            print('Arguments input_dir and output_dir are required for detection')

    if args.command == 'export':
        if not args.input_dir:
            print('input_dir is required for export')
        output_dir = args.output_dir or args.input_dir
        export_results(args.input_dir, output_dir)
