import cv2 as cv
import numpy as np
import pandas as pd
import utils
import os
from typing import Dict, List
import math


class ResultWrapper:
    def __init__(self, params_dict) -> None:
        self.params_dict = params_dict

    @staticmethod
    def from_json(file, images):
        contours = utils.json2contours(file)
        params_dict = get_params_from_contours(contours, images)
        return ResultWrapper(params_dict)

    @staticmethod
    def from_result(results_dict, images):
        params_dict = get_params_from_results(results_dict, images)
        return ResultWrapper(params_dict)

    def get_df(self, fields=('id', 'score', 'a', 'b', 'area')):
        table = []
        for name, params_list in self.params_dict.items():
            for p in params_list:
                row = (name, *tuple(p[k] for k in fields))
                table.append(row)
        return pd.DataFrame(table, columns=['name', *fields])

    def save_annotation(self, file):
        contours_list_dict = {name: [params['cnt'] for params in params_list]
                              for name, params_list
                              in self.params_dict.items()}
        region_data = utils.contours2json(contours_list_dict)
        utils.save_region_data(file, region_data)

    def save_table(self, out_dir, file='results.csv',
                   fields=('id', 'score', 'area', 'perimeter', 'roundness', 'ellipse_area', 'a', 'b', 'a+b', 'on_edge')):
        path = os.path.join(out_dir, file)
        self.get_df(fields).to_csv(path, index=False, float_format='%.3f')

    def visualize(self, images: utils.Images):
        visualize_all_params(images, self.params_dict)


def extract_contour_params(i, cnt, img_shape):
    if len(cnt) < 5:
        return None
    ellipse = cv.fitEllipse(cnt)
    s1 = ellipse[1][0] / 2
    s2 = ellipse[1][1] / 2
    a = max(s1, s2)
    b = min(s1, s2)
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    roundness = (4 * math.pi * area) / (perimeter ** 2)
    params = {
        'id': i + 1,
        'cnt': cnt,
        'ellipse': ellipse,
        'a': a,
        'b': b,
        'area': area,
        'perimeter': perimeter,
        'roundness': roundness,
        'ellipse_area': math.pi * a * b,
        'a+b': a + b,
        'on_edge': utils.on_edge(cnt, img_shape, 15)
    }
    return params


def get_params_from_contours(contours_list_dict, images):
    params_dict = {}
    for name, cnt_list in contours_list_dict.items():
        params_list = []
        img = images[name]
        for i, cnt in enumerate(cnt_list):
            params = extract_contour_params(i, cnt, img)
            if params:
                params_list.append(params)
        params_dict[name] = params_list
    return params_dict


def get_params_from_result(result, img: np.ndarray):
    params_list = []
    for i in range(result['masks'].shape[2]):
        mask = result['masks'][:, :, i].astype(np.uint8)
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        cnt = contours[0]
        params = extract_contour_params(i, cnt, img.shape)
        if not params:
            continue
        ext_params = {
            'score': result['scores'][i],
            'box': result['rois'][i],
            'mask': mask
        }
        params.update(ext_params)
        params_list.append(params)
    return params_list


def get_params_from_results(results_dict, images: utils.Images) -> Dict:
    params_dict = {}
    for name, result in results_dict.items():
        params_dict[name] = get_params_from_result(result, images[name])
    return params_dict


def visualize_params(img, params_list):
    k = max(img.shape) / 1024
    t = round(k)
    scale = k * 1.2
    for params in params_list:
        # contours
        e_color, cnt_color = (0, 0, 255), (0, 255, 0)
        if params['on_edge']:
            e_color, cnt_color = (0, 0, 130), (0, 130, 0)

        cv.ellipse(img, params['ellipse'], e_color, t, cv.LINE_AA)
        cv.drawContours(img, [params['cnt']], -1, cnt_color, t, cv.LINE_AA)
        # text
        y1, x1, y2, x2 = params['box']
        text = '{id} {score:.2f}'.format(**params)
        font = cv.FONT_HERSHEY_PLAIN
        cv.putText(img, text, (x1 + 1, y1 + 1), font, scale, (0, 0, 0), 2 * t, cv.LINE_AA)
        cv.putText(img, text, (x1, y1), font, scale, (255, 255, 255), t, cv.LINE_AA)


def visualize_all_params(images_dict: utils.Images, params_dict):
    for name, params_dict in params_dict.items():
        img = images_dict[name]
        visualize_params(img, params_dict)


def get_params_and_visualize(images_dict: utils.Images, results_dict):
    params_dict = get_params_from_results(results_dict, images_dict)
    visualize_all_params(images_dict, params_dict)
