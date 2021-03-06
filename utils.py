import json
import math
import os
import shutil
from pathlib import Path
from typing import List, Dict

import cv2 as cv
import numpy as np

Images = Dict[str, np.ndarray]


def load_images(input_dir, extensions=('png', 'jpg', 'tif')) -> Images:
    images = {}
    files = os.listdir(input_dir)
    for file in files:
        ext = file.split('.')[-1]
        if ext not in extensions:
            continue
        path = os.path.join(input_dir, file)
        img = cv.imread(path, cv.IMREAD_COLOR + cv.IMREAD_ANYDEPTH)
        if img.dtype is np.dtype('uint16'):
            img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        images[file] = img
    return images


def resize(image: np.ndarray, size=1024) -> np.ndarray:
    h = image.shape[0]
    w = image.shape[1]
    ratio = size / max(h, w)
    return cv.resize(image, None, fx=ratio, fy=ratio)


def resize_images(images: Images, size=1024) -> Images:
    return {name: resize(img, size) for name, img in images.items()}


def write_images(images: Images, output_dir, ext='png', makedirs=True):
    if not os.path.exists(output_dir) and makedirs:
        os.makedirs(output_dir, exist_ok=True)
    for name, image in images.items():
        name, old_ext = name.rsplit('.', 1)
        new_name = f'{name}.{ext}'
        out_path = os.path.join(output_dir, new_name)
        cv.imwrite(out_path, image)


def prepare_images(input_dir, output_dir, ext='png', size=1024):
    images = load_images(input_dir)
    images = resize_images(images, size)
    write_images(images, output_dir, ext)


def show_images(images: Images):
    for name, img in images.items():
        cv.imshow(name, img)
        cv.waitKey(0)


def contour_to_shape(contour):
    shape = {
        "name": "polygon",
        "all_points_x": [],
        "all_points_y": [],
    }

    for chunk in contour:
        point = chunk[0]
        x = int(point[0])
        y = int(point[1])
        shape['all_points_x'].append(x)
        shape['all_points_y'].append(y)

    return shape


def contour2region(contour):
    return {
        "shape_attributes": contour_to_shape(contour),
        "region_attributes": {}
    }


def box_to_shape(box):
    return {
        "name": "rect",
        "x": box[0],
        "y": box[1],
        "width": box[2],
        "height": box[3]
    }


def export_shape_csv(output_file, shapes_dict, convert_f):
    header = "filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n"
    lines = [header]
    for file in shapes_dict:
        shapes = shapes_dict[file]
        for i, shape_obj in enumerate(shapes):
            shape = convert_f(shape_obj)
            shape_str = json.dumps(shape)
            shape_str = shape_str.replace('"', '""')
            region_count = len(shapes)
            line = '{},{},{},{},{},"{}",{}\n'.format(file, 0, '"{}"', region_count, i, shape_str, '"{}"')
            lines.append(line)
    with open(output_file, 'w') as file:
        file.writelines(lines)


def export_contours_csv(output_file, contours_dict):
    export_shape_csv(output_file, contours_dict, contour_to_shape)


def export_boxes_csv(output_file, boxes_dict):
    export_shape_csv(output_file, boxes_dict, box_to_shape)


def mark2cnt(img, mark_color):
    mask = cv.inRange(img, mark_color, mark_color)
    _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def mark_to_boxes(img, mark_color):
    mask = cv.inRange(img, mark_color, mark_color)
    _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        boxes.append(cv.boundingRect(contour))
    return boxes


def get_mask(shape, contours, value, dtype=np.uint8):
    mask = np.zeros(shape, dtype=dtype)
    color = tuple([value] * shape[2])
    for cnt in contours:
        cv.fillPoly(mask, [cnt], color)
    return mask


def iou(shape, contours1, contours2):
    mask1 = get_mask(shape, contours1, 1)
    mask2 = get_mask(shape, contours2, 1)
    inter = mask1 * mask2
    union = (mask1 + mask2) / 2
    return np.sum(inter) / np.sum(union)


def create_circle_contour(r):
    d = 2 * r
    mask = np.zeros((d, d, 1), dtype=np.uint8)
    cv.circle(mask, (r, r), r, 250)
    _, cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return cnts[0]


def json2contours(file):
    f = open(file)
    region_data = json.load(f)
    contours_dict = {}
    for key, data in region_data.items():
        name = data['filename']
        regions = data['regions']
        contours = []
        for region in regions:
            attrs = region['shape_attributes']
            points_x = attrs['all_points_x']
            points_y = attrs['all_points_y']
            polygon = np.array(list(zip(points_x, points_y)))
            polygon.resize((len(points_x), 1, 2))
            # print('polygon: {}\n cnt: {}'.format(polygon, cnt))
            contours.append(polygon)
        # ext = name.split('.')[-1]
        # base_name = name.replace('.' + ext, '')
        contours_dict[name] = contours
    return contours_dict


def json2masks(file, imgs_dir, masks_dir, postfix='_mask'):
    f = open(file)
    region_data = json.load(f)
    masks = {}
    for key in region_data:
        name = region_data[key]['filename']
        img_path = os.path.join(imgs_dir, name)
        img = cv.imread(img_path, 1)
        mask = np.zeros((*img.shape[:2], 1), np.uint8)
        regions = region_data[key]['regions']
        for region in regions:
            attrs = region['shape_attributes']
            points_x = attrs['all_points_x']
            points_y = attrs['all_points_y']
            polygon = np.array(list(zip(points_x, points_y)))
            polygon.resize((len(points_x), 1, 2))
            cv.fillConvexPoly(mask, polygon, 255)
        masks[name] = mask
        ext = name.split('.')[-1]
        base_name = name.replace('.' + ext, '')
        path = os.path.join(masks_dir, '{}{}.{}'.format(base_name, postfix, ext))
        cv.imwrite(path, mask)
    return masks


def json2separated_masks(file, imgs_dir, masks_dir, postfix='_mask'):
    f = open(file)
    region_data = json.load(f)
    masks = {}
    for key in region_data:
        name = region_data[key]['filename']
        img_path = os.path.join(imgs_dir, name)
        img = cv.imread(img_path, 1)
        mask = np.zeros((*img.shape[:2], 1), np.uint8)
        regions = region_data[key]['regions']
        for i, region in enumerate(regions):
            attrs = region['shape_attributes']
            points_x = attrs['all_points_x']
            points_y = attrs['all_points_y']
            polygon = np.array(list(zip(points_x, points_y)))
            polygon.resize((len(points_x), 1, 2))
            cv.fillConvexPoly(mask, polygon, (i + 1) * 20)
        masks[name] = mask
        ext = name.split('.')[-1]
        base_name = name.replace('.' + ext, '')
        path = os.path.join(masks_dir, '{}{}.{}'.format(base_name, postfix, ext))
        cv.imwrite(path, mask)
    return masks


def contours2json(contours_list_dict):
    region_data = {}
    for name, contours_list in contours_list_dict.items():
        regions = []
        for cnt in contours_list:
            epsilon = 1.2
            approx = cv.approxPolyDP(cnt, epsilon, True)
            regions.append(contour2region(approx))
        region_data[name + '0'] = {
            "filename": name,
            "size": "0",
            "regions": regions,
            "file_attributes": {}
        }
    return region_data


def masks2json(masks_list_dict):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    contours_list_dict = {}
    for name, mask_list in masks_list_dict.items():
        contours_list = []
        for mask in mask_list:
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
            if not contours:
                continue
            for cnt in contours:
                epsilon = 1.2
                approx = cv.approxPolyDP(cnt, epsilon, True)
                contours_list.append(approx)
        contours_list_dict[name] = contours_list
    return contours2json(contours_list_dict)


def save_region_data(file, region_data):
    f = open(file, 'w+')
    json.dump(region_data, f)


def image_masks2json(json_file, masks_dir, extensions=('png', 'tif', 'jpg')):
    masks = {}
    for file in os.listdir(masks_dir):
        ext = file.split('.')[-1]
        if ext not in extensions:
            continue
        path = os.path.join(masks_dir, file)
        mask = cv.imread(path, 0)
        masks[file] = extract_masks(mask)
    region_data = masks2json(masks)
    save_region_data(json_file, region_data)


def merge_data(in_dir: str, out_dir: str):
    """
    Объединяет изображения и файл разметки
    :param in_dir: директория, которая содержит вложенные директории с изображениями и разметкой
    :param out_dir: директория в которую будут записаны изображения из поддиректорий
    и общий файл с разметкой via_region_data.json
    """

    out_path = Path(out_dir)
    if not out_path.exists():
        out_path.mkdir()

    final_region_data = {}

    i = 0
    for path in Path(in_dir).iterdir():
        if not path.is_dir():
            continue
        region_data_path = path / 'via_region_data.json'
        region_data = json.load(region_data_path.open())
        for key, value in region_data.items():
            filename = value['filename']
            name, ext = filename.rsplit('.', 1)
            print(key, filename)
            new_filename = '{}.{}'.format(i, ext)
            new_key = new_filename + str(value['size'])
            print(new_key, new_filename)
            value['filename'] = new_filename
            final_region_data[new_key] = value
            # копирование файла с новым именем
            shutil.copy(str(path / filename), str(out_path / new_filename))
            i += 1
    json.dump(final_region_data, (out_path / 'via_region_data.json').open('w'))


def extract_masks(mask) -> List:
    """
    Выделяет бинарные маски из монолитной серой маски
    :param mask: серая маска в которой значение пикселя соответствует классу
    :return: список бинарных масок
    """
    masks = []
    classes = set()
    for row in mask:
        for v in row:
            classes.add(v)
    classes.remove(0)  # удаляем фон
    for c in classes:
        m = mask.copy()
        m[m != c] = 0
        m[m == c] = 255
        masks.append(m)
    return masks


def change_ext(images: Images, new_ext) -> Images:
    return {f"{name.rsplit('.', 1)[0]}.{new_ext}": img for name, img in images.items() if '.' in name}


def poly_from_str(s):
    xs, ys = [], []
    for str_point in s.split(';'):
        x, y = str_point.split(',')
        xs.append(float(x))
        ys.append(float(y))
    return np.array(xs), np.array(ys)


def on_edge(cnt, img_shape, radius) -> bool:
    h, w = img_shape[:2]
    points_on_edge = 0
    for point in cnt:
        x, y = point[0]
        if x < radius or y < radius or x > w - radius or y > h - radius:
            points_on_edge += 1
    print(points_on_edge)
    return points_on_edge > 0


def roundness(cnt) -> float:
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    if perimeter > 0:
        return (4 * math.pi * area) / (perimeter ** 2)
    else:
        return 0


def get_contours(result: dict) -> list:
    masks = result['masks']
    contours = []
    for i in range(masks.shape[2]):
        mask = masks[:, :, i].astype(np.uint8)
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        if contours:
            cnt = contours[0]
            contours.append(cnt)
    return contours


def put_text(img, text, x, y):
    k = max(img.shape) / 1024
    t = round(k)
    scale = k * 1.2
    font = cv.FONT_HERSHEY_PLAIN
    cv.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), 2 * t, cv.LINE_AA)
    cv.putText(img, text, (x, y), font, scale, (255, 255, 255), t, cv.LINE_AA)


def put_scores(img, result):
    for box, score in zip(result['rois'], result['scores']):
        y, x, _, _ = box
        put_text(img, f'{score:.2f}', x, y)


def draw_masks_contours(img, masks, color):
    k = max(img.shape) / 1024
    t = round(k)
    n = masks.shape[2]
    for i in range(n):
        mask = masks[:, :, i]
        _, contours, _ = cv.findContours(mask.copy().astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        cv.drawContours(img, contours, -1, color, t, cv.LINE_AA)


def visualize_result(img, result: dict, gt_masks=None, scores=False):
    masks = result['masks']
    if gt_masks is not None:
        draw_masks_contours(img, gt_masks, (0, 0, 255))
    draw_masks_contours(img, masks, (0, 255, 0))
    if scores:
        put_scores(img, result=result)


def get_bin_mask(mask):
    n = np.max(mask)
    bin_mask = np.zeros((*mask.shape[:2], n), dtype=np.bool)
    for i in range(1, n + 1):
        bin_mask[:, :, i - 1] = mask == i
    return bin_mask


def visualise(img, mask_gt, mask_det):
    draw_masks_contours(img, mask_gt, (0, 255, 0))
    draw_masks_contours(img, mask_det, (0, 0, 255))


def f_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def avg_roundness(contours) -> float:
    if len(contours) > 0:
        return sum([roundness(cnt) for cnt in contours]) / len(contours)
    else:
        return 0


def get_cnt_from_mask(mask):
    mask = mask.astype(np.uint8)
    _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
    return contours[0]


def bin_mask_to_list(bin_mask):
    return [bin_mask[:, :, i] for i in range(bin_mask.shape[2])]


def mask_area(mask) -> float:
    return np.count_nonzero(mask)


def area_filter(gt_boxes, gt_class_ids, gt_masks, results, min_area, max_area):
    new_boxes, new_classes, new_masks, new_results = [], [], [], []
    for img_boxes, img_classes, img_masks, res in zip(gt_boxes, gt_class_ids, gt_masks, results):
        # Filter GT
        gt_index = [min_area < mask_area(mask) < max_area for mask in bin_mask_to_list(img_masks)]
        new_boxes.append(img_boxes[gt_index])
        new_classes.append(img_classes[gt_index])
        new_masks.append(img_masks[:, :, gt_index])

        # Filter result
        res_index = [min_area < mask_area(mask) < max_area for mask in bin_mask_to_list(res['masks'])]
        new_results.append({
            'rois': res['rois'][res_index],
            'class_ids': res['class_ids'][res_index],
            'scores': res['scores'][res_index],
            'masks': res['masks'][:, :, res_index]
        })

    return new_boxes, new_classes, new_masks, new_results
