FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN useradd -ms /bin/bash user

WORKDIR /app
RUN chown user: /app && chmod u+w /app

RUN apt-get update && apt-get install -y git libsm6 libxext6 libxrender1

COPY app_requirements.txt /app/
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r app_requirements.txt

COPY *.py /app/
COPY static /app/static
COPY templates /app/templates

RUN chmod a+x vesicle.py detector.py

USER user

CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:8000"]