FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN useradd -ms /bin/bash user

WORKDIR /app
RUN chown user: /app
RUN chmod u+w /app

RUN apt-get update && apt-get install -y git libsm6 libxext6 libxrender1

COPY train_requirements.txt /app/
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r train_requirements.txt

COPY models/final.h5 /app/models/
COPY models/mask_rcnn_coco.h5 /app/

COPY *.py /app/
RUN chmod a+x vesicle.py detector.py

USER user

ENTRYPOINT ["./detector.py"]

