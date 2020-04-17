from flask import Flask, escape, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
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
                image = cv.imdecode(img_array, 0)
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
    path = os.path.join(UPLOAD_FOLDER, session_id)
    out_path = os.path.join(path, 'result')
    utils.write_images(images, path)
    vesicle.detect_from_dir(path, out_path, MODEL_PATH)
    return redirect(url_for('result'))


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run()
