import cv2 as cv
import os
import json


def load_images(input_dir):
    images = {}
    files = os.listdir(input_dir)
    for file in files:
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


def export_contours_csv(output_file, contours_dict):
    header = "filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n"
    lines = [header]
    for file in contours_dict:
        contours = contours_dict[file]
        for i, contour in enumerate(contours):
            shape = contour_to_shape(contour)
            shape_str = json.dumps(shape)
            shape_str = shape_str.replace('"', '""')
            region_count = len(contours)
            line = '{},{},{},{},{},"{}",{}\n'.format(file, 0, '"{}"', region_count, i, shape_str, '"{}"')
            lines.append(line)
    with open(output_file, 'w') as file:
        file.writelines(lines)
