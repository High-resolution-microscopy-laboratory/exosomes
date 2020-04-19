from flask import Flask, escape, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from shutil import make_archive, move
import uuid
from typing import Dict, List
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

model = vesicle.load_model(MODEL_PATH)
model.keras_model._make_predict_function()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_images(cur_request) -> Images:
    images = {}
    for files in cur_request.files.listvalues():
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv.imdecode(img_array, 1)
                images[filename] = image
    return images


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    session_id = uuid.uuid4().hex
    images = get_images(request)
    images = utils.resize_images(images, size=1024)
    images = utils.change_ext(images, IMG_EXT)

    path = os.path.join(UPLOAD_FOLDER, session_id)
    out_path = os.path.join(path, 'result')

    vesicle.detect(images, out_path, model)
    utils.write_images(images, path, ext=IMG_EXT)

    make_archive('result', 'zip', out_path)
    move('result.zip', path)
    return redirect(url_for('result', result_id=session_id))


def is_img_path(path) -> bool:
    if '.' in path:
        return path.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return False


def get_result_dir(result_id: str) -> str:
    return os.path.join(UPLOAD_FOLDER, result_id)


@app.route('/uploads/<result_id>/<file>')
def get_file(result_id, file):
    directory = os.path.join(UPLOAD_FOLDER, result_id)
    return send_from_directory(directory, file)


@app.route('/result/<result_id>')
def result(result_id):
    directory = os.path.join(UPLOAD_FOLDER, result_id)
    files = [p for p in os.listdir(directory) if is_img_path(p)]
    return render_template('result.html', result_id=result_id, files=files)


@app.route('/download/<result_id>')
def download(result_id: str):
    return send_from_directory(get_result_dir(result_id), 'result.zip')


if __name__ == '__main__':
    app.run()
