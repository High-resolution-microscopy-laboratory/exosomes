import cv2 as cv
import numpy as np
import pandas as pd
import utils


class ResultWrapper:
    def __init__(self, results_dict) -> None:
        self.results = results_dict
        self.params_dict = get_params_from_results(results_dict)

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

    def save_table(self, file):
        writer = pd.ExcelWriter(file)
        self.get_df().to_excel(writer)


def get_params_from_result(result):
    params_list = []
    for i in range(result['masks'].shape[2]):
        mask = result['masks'][:, :, i].astype(np.uint8)
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        cnt = contours[0]
        if len(cnt) < 5:
            continue
        ellipse = cv.fitEllipse(cnt)
        s1 = ellipse[1][0] / 2
        s2 = ellipse[1][1] / 2
        params = {
            'id': i,
            'cnt': cnt,
            'score': result['scores'][i],
            'ellipse': ellipse,
            'a': max(s1, s2),
            'b': min(s1, s2),
            'area': cv.contourArea(cnt),
            'box': result['rois'][i],
            'mask': mask
        }
        params_list.append(params)
    return params_list


def get_params_from_results(results_dict):
    params_dict = {}
    for name, result in results_dict.items():
        params_dict[name] = get_params_from_result(result)
    return params_dict


def visualize_params(img, params_list):
    k = max(img.shape) / 1024
    t = round(k)
    scale = k * 1.2
    for params in params_list:
        # contours
        cv.ellipse(img, params['ellipse'], (0, 0, 255), t, cv.LINE_AA)
        cv.drawContours(img, [params['cnt']], -1, (0, 255, 0), t, cv.LINE_AA)
        # text
        y1, x1, y2, x2 = params['box']
        text = '{id} {score:.2f}'.format(**params)
        font = cv.FONT_HERSHEY_PLAIN
        cv.putText(img, text, (x1 + 1, y1 + 1), font, scale, (0, 0, 0), 2 * t, cv.LINE_AA)
        cv.putText(img, text, (x1, y1), font, scale, (255, 255, 255), t, cv.LINE_AA)


def visualize_all_params(images_dict, params_dict):
    for name, params_dict in params_dict.items():
        img = images_dict[name]
        visualize_params(img, params_dict)


def get_params_and_visualize(images_dict, results_dict):
    params_dict = get_params_from_results(results_dict)
    visualize_all_params(images_dict, params_dict)
