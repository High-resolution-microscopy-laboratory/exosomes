import os
import uuid
from shutil import make_archive, move
from typing import Dict

import cv2 as cv
import numpy as np
from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import utils
import vesicle

Images = Dict[str, np.ndarray]

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tif', 'tiff'}
MODEL_PATH = 'models/mask_rcnn_vesicle.h5'
IMG_EXT = 'jpg'
MAX_UPLOADS = 10
ANNOTATION_FILE_NAME = 'via_region_data_detect.json'

model = vesicle.load_model(MODEL_PATH)
model.keras_model._make_predict_function()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_num_uploads(cur_request) -> int:
    n = 0
    for files in cur_request.files.listvalues():
        for file in files:
            if file:
                n += 1
    return n


def invalid_extension(cur_request) -> bool:
    for files in cur_request.files.listvalues():
        for file in files:
            if file and not allowed_file(file.filename):
                return True
    return False


def get_images(cur_request) -> Images:
    images = {}
    for files in cur_request.files.listvalues():
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR + cv.IMREAD_ANYDEPTH)
                if image.dtype is np.dtype('uint16'):
                    image = cv.normalize(image, image, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                images[filename] = image
    return images


def get_result_dir(result_id: str) -> str:
    return os.path.join(UPLOAD_FOLDER, result_id)


def get_visualised_imgs_dir(result_id: str) -> str:
    return os.path.join(UPLOAD_FOLDER, result_id, 'visualised')


@app.route('/')
def main():
    return render_template('index.html', max_uploads=MAX_UPLOADS)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    num_uploads = get_num_uploads(request)
    if num_uploads == 0:
        return redirect(url_for('error', status='empty'))

    if num_uploads > MAX_UPLOADS:
        return redirect(url_for('error', status='limit_exceed'))

    if invalid_extension(request):
        return redirect(url_for('error', status='invalid_type'))

    session_id = uuid.uuid4().hex
    images = get_images(request)
    images = utils.change_ext(images, IMG_EXT)

    base_path = os.path.join(UPLOAD_FOLDER, session_id)
    prepared_img_path = os.path.join(base_path, 'prepared')
    visualised_img_path = get_visualised_imgs_dir(session_id)
    result_path = os.path.join(base_path)

    utils.write_images(images, prepared_img_path, ext=IMG_EXT)
    vesicle.detect(images, result_path, model)
    utils.write_images(images, visualised_img_path, ext=IMG_EXT)

    make_archive('result', 'zip', base_path)
    move('result.zip', base_path)
    return redirect(url_for('result', result_id=session_id))


@app.route('/uploads/<result_id>/<file>')
def get_visualised_img(result_id, file):
    directory = get_visualised_imgs_dir(result_id)
    return send_from_directory(directory, file)


@app.route('/result/<result_id>')
def result(result_id):
    directory = os.path.join(UPLOAD_FOLDER, result_id, 'visualised')
    if not os.path.exists(directory):
        abort(404)
    files = [p for p in sorted(os.listdir(directory)) if allowed_file(p)]
    return render_template('result.html', result_id=result_id, files=files)


@app.route('/download/<result_id>')
def download(result_id: str):
    return send_from_directory(get_result_dir(result_id), 'result.zip')


@app.route('/error/<status>')
def error(status: str):
    if status == 'empty':
        return render_template('error.html', title='Empty upload',
                               description=f'')
    if status == 'limit_exceed':
        return render_template('error.html', title='File limit exceeded',
                               description=f'You can upload not more then {MAX_UPLOADS} images at a time')
    if status == 'invalid_type':
        return render_template('error.html', title='Invalid file format',
                               description='Supported image formats: png, jpg, jpeg, tif, tiff')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', title='Page not found'), 404


if __name__ == '__main__':
    app.run()
