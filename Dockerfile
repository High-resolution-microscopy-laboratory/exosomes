FROM tensorflow/tensorflow:1.11.0-py3

RUN useradd -ms /bin/bash user

WORKDIR /app
RUN chown user: /app
RUN chmod u+w /app

COPY requirements.txt /app/
COPY *.py /app/

RUN chmod a+x vesicle.py

COPY models/final.h5 /app/models/

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y libsm6 libxext6 libxrender1

RUN pip install -r requirements.txt

USER user

ENTRYPOINT ["./vesicle.py"]

