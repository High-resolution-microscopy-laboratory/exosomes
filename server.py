from flask import Flask, escape, request, render_template, redirect, url_for, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
from shutil import make_archive, move
import uuid
from typing import Dict, List, Tuple
import cv2 as cv
import numpy as np

import utils
import vesicle

Images = Dict[str, np.ndarray]

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tif', 'tiff'}
MODEL_PATH = 'models/final.h5'
IMG_EXT = 'png'
MAX_UPLOADS = 5

model = vesicle.load_model(MODEL_PATH)
model.keras_model._make_predict_function()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_img_path(path) -> bool:
    if '.' in path:
        return path.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return False


def exceed_num_uploads(cur_request) -> bool:
    n = 0
    for files in cur_request.files.listvalues():
        n += len(files)
    return n > MAX_UPLOADS


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
                image = cv.imdecode(img_array, 1)
                images[filename] = image
    return images


def get_result_dir(result_id: str) -> str:
    return os.path.join(UPLOAD_FOLDER, result_id)


def get_visualised_imgs_dir(result_id: str) -> str:
    return os.path.join(UPLOAD_FOLDER, result_id, 'visualised')


@app.route('/')
def main():
    return render_template('index.html', max_uploads=MAX_UPLOADS)


@app.route('/upload', methods=['POST'])
def upload():
    if exceed_num_uploads(request):
        return render_template('error.html', title='File limit exceeded',
                               description=f'You can upload not more then {MAX_UPLOADS} images at a time')

    if invalid_extension(request):
        return render_template('error.html', title='Invalid file format',
                               description='Supported image formats: png, jpg, jpeg, tif, tiff')

    session_id = uuid.uuid4().hex
    images = get_images(request)
    images = utils.resize_images(images, size=1024)
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
    files = [p for p in os.listdir(directory) if is_img_path(p)]
    return render_template('result.html', result_id=result_id, files=files)


@app.route('/download/<result_id>')
def download(result_id: str):
    return send_from_directory(get_result_dir(result_id), 'result.zip')


@app.route('/help')
def help():
    return render_template('help.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', title='Page not found'), 404


if __name__ == '__main__':
    app.run()
