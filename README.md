# Automatic License Plate Recognition (ALPR)

### Описание проекта

Основаня цель проекта - детекция и распознавание автомобильных номеров.

### Инференс

В ноутбуке inference_notebook.ipynb написан поэтапный запуск инференса ALPR.
В качестве модели для детекции могут быть использованы FasterRCNN или Yolov5.
Для распознавания номеров используется связка ResNet + LSTM + CTC из библиотеки
[EasyOCR](https://github.com/JaidedAI/EasyOCR).

Скорость инференса ~ 10-11 изображений в секунду.

Для запуска FasterRCNN необходимо добавить веса в папку DetactionModels. 
Веса можно скачать по ссылке: https://disk.yandex.ru/d/NlvmdsBvbNbltQ .

### Метрики моделей

FasterRCNN

| Model      | mAP 0.5 | mAP 0.5_0.95 | Precision | Recall |
|:-----------|:--------|:-------------|:----------|:-------|
| FasterRCNN | 0.946   | 0.78         | 0.907     | 0.989  |


Метрики обучения на каждой эпохе можно посмотреть в 
[faster_rcnn_v5.txt](Logs/faster_rcnn_v5.txt).

EasyOCR

| Model   | Best_accuracy | Best_norm_ED |
|:--------|:--------------|:-------------|
| EasyOCR | 95.927        | 0.9905       | 



Метрики обучения на каждой эпохе можно посмотреть в 
[log_train.py](easyocr_trainer/saved_models/alpr_new_filtered/log_train.txt).

### Датасаеты

Для обучения использовались следующие датасеты:

1. Kaggle - VKCV 2022 Contest_02
2. Kaggle - Nomeroff Russian License plates



