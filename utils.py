import cv2 as cv
import os
import json
import numpy as np


def load_images(input_dir, extensions=['png']):
    images = {}
    files = os.listdir(input_dir)
    for file in files:
        ext = file.split('.')[-1]
        if ext not in extensions:
            continue
        path = os.path.join(input_dir, file)
        img = cv.imread(path, 1)
        images[file] = img
    return images


def prepare_images(input_dir, output_dir, ext='.png', size=1024):
    images = load_images(input_dir)
    for name in images:
        image = images[name]
        h = image.shape[0]
        w = image.shape[1]
        ratio = size / max(h, w)
        resized = cv.resize(image, None, fx=ratio, fy=ratio)
        new_name = name.replace('.tif', ext)
        out_path = os.path.join(output_dir, new_name)
        cv.imwrite(out_path, resized)


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


def masks2json(json_file, masks_dir, extensions=['png']):
    region_data = {}
    for file in os.listdir(masks_dir):
        ext = file.split('.')[-1]
        if ext not in extensions:
            continue
        path = os.path.join(masks_dir, file)
        mask = cv.imread(path, 0)
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        regions = []
        for cnt in contours:
            epsilon = 0.01 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            regions.append(contour2region(approx))
        region_data[file + '0'] = {
            "filename": file,
            "size": "0",
            "regions": regions,
            "file_attributes": {}
        }
    f = open(json_file, 'w+')
    json.dump(region_data, f)
