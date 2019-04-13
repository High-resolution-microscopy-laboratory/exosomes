from utils import merge_data, json2masks, json2separated_masks
import augment
import shutil
from pathlib import Path

NEW_DATA_DIR = 'data/new'
OUT_DIR = 'data/train'
MASK_DIR = 'data/masks'
TMP_DIR = 'data/tmp'

if __name__ == '__main__':
    # Подготовка и очистка директорий
    dirs = [OUT_DIR, MASK_DIR, TMP_DIR]
    for d in dirs:
        if Path(d).exists():
            shutil.rmtree(d)
        Path(d).mkdir()
    # Объединение данных из разных папок
    merge_data(NEW_DATA_DIR, TMP_DIR)
    # Создание масок
    json2separated_masks(TMP_DIR + '/via_region_data.json', TMP_DIR, MASK_DIR, postfix='')
    # Аугментации
    augment.apply(TMP_DIR, MASK_DIR, OUT_DIR, n=3)

