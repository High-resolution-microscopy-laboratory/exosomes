## Описание
Готовый функционал:

1. Сегментация везикул
2. Экспорт в формат для разметки [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/)
3. Экспорт таблицы c параметрами везикул

## Установка

Можно установить необходимые зависимости через pip или использовать собранный docker контейнер
### Python
```
pip install -r requirements.txt
```
Для работы нужно [скачать](https://bitbucket.org/vergilius/vesicles/downloads/)
и сохранить веса обученной модели в корне проекта в директории `models`

### Docker

https://docs.docker.com/install/

`docker pull exactly/vesicles`

## Использование
### Запуск детектора

`python vesicle.py detect --input_dir path/to/images/ --output_dir path/to/output/`

или

`vesicle.py detect -i path/to/images/ -o path/to/output/`

`--input_dir` - директория с исходными изображениями

`--output_dir` - директория в которую будут записаны предобработанные изображения и файл разметки

После завершения работы детектора в `--output_dir` будет доступен файл разметки `via_region_data.json`
его можно загрузить в [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/) для просмотра и исправления

Исправленный файл нужно экспортировать и сохранить `Annotation -> Export Annotations (as json)`


### Экспорт результатов

`python vesicle.py export --input_dir path/to/images/ --output_dir path/to/output/`

или

`vesicle.py export -i path/to/images/ -o path/to/output/`

`--input_dir` - директория содержащая исправленный `via_region_data.json`

`--output_dir` - директория в которую будет записана таблица с результатами `results.csv`

Если не было ручных исправлений можно не экспортировать файл с разметкой и просто указать директорию с `via_region_data_detect.json` в качестве любого из параметров `--input_dir` или `--output_dir` (они будут совпадать и второй можно не указывать)

Например

`vesicle.py export -input_dir path/to/output`

или

`vesicle.py export -i path/to/output`

### Docker

`vesicle.sh <command> <root> <input_dir> <output_dir>`

Пример:

`/home/user/images_dir/` - тут лежат изображения

`/home/user/results_dir/` - сюда будет записан результат

`vesicle.sh detect /home/user/ images_dir results_dir`

`vesicle.sh export results_dir`


### Структура директории с результатами

```
output_dir
├── 01_100k.png
├── 02_100k.png
├── 03_100k.png
├── results.csv                    - таблица с результатами
├── via_region_data_detect.json    - файл с разметкой детектора
└── via_region_data.json           - исправленный финальный файл с разметкой
```

