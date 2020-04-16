from flask import Flask, escape, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import uuid
import cv2 as cv

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tif', 'tiff'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    request.get_data()
    result_id = uuid.uuid4().hex

    for files in request.files.listvalues():
        for file in files:
            if not allowed_file(file.filename):
                return 'Invalid file extension', 400
            if file:
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, result_id)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                file.save(os.path.join(path, filename))
    return redirect(url_for('result'))


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run()
