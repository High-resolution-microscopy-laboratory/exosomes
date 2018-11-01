## Описание
Готовый функционал:
1. Сегментация везикул
2. Экспорт в формат для разметки [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/)
3. Экспорт таблицы c параметрами везикул

## Установка
Нужно установить Mask RCNN по этой инструкции:
https://github.com/matterport/Mask_RCNN#installation


```
pip install -r requirements.txt
```
Для работы нужно [скачать](https://bitbucket.org/vergilius/vesicles/downloads/) и сохранить веса обученной модели в корне проекта в директории `models`

## Использование
Запуск детектора

`python vesicle.py detect --input_dir=path/to/images/ --output_dir=path/to/output/`

Экспорт параметров

`python vesicle.py export --input_dir=path/to/images/ --output_dir=path/to/output/`