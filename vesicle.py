#!/usr/bin/env python3

import os
import shutil
import result

DEFAULT_FILE_NAME = 'via_region_data.json'
DETECTOR_FILE_NAME = 'via_region_data_detect.json'


def detect(input_dir, output_dir, weights_path):
    # Mask RCNN
    import tensorflow as tf
    import mrcnn.model as modellib

    import detector
    import utils

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
        description='Train Mask R-CNN to detect vesicle')
    parser.add_argument('command',
                        metavar="'<command>'",
                        help="'detect' or 'export'")
    parser.add_argument('--input_dir', '-i', required=False,
                        metavar='/path/to/input/images/',
                        help='Directory of the input images')
    parser.add_argument('--output_dir', '-o', required=False,
                        metavar='/path/to/output/dir/',
                        help='Directory for the output data')
    parser.add_argument('--weights', required=False,
                        default='models/final.h5',
                        metavar='/path/to/model/',
                        help='Model weights path (default=models/final.h5)')
    args = parser.parse_args()

    if args.command == 'detect':
        if args.input_dir and args.output_dir:
            detect(args.input_dir, args.output_dir, args.weights)
        else:
            print('Arguments --input_dir and --output_dir are required for detection')

    if args.command == 'export':
        if args.input_dir or args.output_dir:
            # Один из аргументов можно опустить
            input_dir = args.input_dir or args.output_dir
            output_dir = args.output_dir or args.input_dir
            is_same_dir = input_dir == output_dir
            # print(is_same_dir)
            # input_file_name = DETECTOR_FILE_NAME if is_same_dir else DETECTOR_FILE_NAME
            export_results(input_dir, output_dir)
        else:
            print('--input_dir or --output_dir are required for export')
