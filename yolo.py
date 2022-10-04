import cv2
import numpy as np
import imutils
import easyocr
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet
from matplotlib import pyplot as pl

# Загрузка библиотек, изображений, данных
nPlateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 200
color = (255,0,255)
# Задание порога NMS Threshold
score_threshold = 0.6
# Задание порога IoU threshold
iou_threshold = 0.4
cfg_file = "cfg/yolov3.cfg"
# Загрузка весов модели YOLO на датасете COCO
weight_file = "weights/yolov3.weights"
namesfile = "data/coco.names"
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
# Загрузка изображений, задание списка номеров для удаления
del_number = ['WAX1267']
img_backgr = cv2.imread("images/img_backgr3.jpeg")
img_backgr = cv2.cvtColor(img_backgr, cv2.COLOR_BGR2RGB)
original_image = cv2.imread("images/parking3.jpeg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

image = cv2.resize(original_image, (int(original_image.shape[1] / 5), int(original_image.shape[0] / 5)))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Функция для извлечения координат Bounding boxes
def coord(box):
    l = original_image.shape[1]
    h = original_image.shape[0]
    y1 = abs(int(h * (box[1].item() - 0.5 * box[3].item())))
    y2 = abs(int(h * (box[1].item() + 0.5 * box[3].item())))
    x1 = abs(int(l * (box[0].item() - 0.5 * box[2].item())))
    x2 = abs(int(l * (box[0].item() + 0.5 * box[2].item())))
    return y1, y2, x1, x2

# Детекция объектов с помощью YOLO
img = cv2.resize(original_image, (m.width, m.height))
boxes = detect_objects(m, img, iou_threshold, score_threshold)

# Распознавание номеров в полученных Bounding box
# индексы bounding boxes для удаления из списка
del_list = []

for i in range(len(boxes)):
    y1, y2, x1, x2 = coord(boxes[i][:])
    img_crop = original_image[y1:y2, x1:x2]
    imgGray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)

    for (x, y, w, h) in numberPlates:
        area = 160 * 50
        if area > minArea:
            cv2.rectangle(img_crop, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img_crop, "Number Plate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgRoi = img_crop[y:y + h, x:x + w]
            # показ вырезанного bounding box
            # pl.imshow(cv2.cvtColor(imgRoi, cv2.COLOR_BGR2RGB))
            # pl.show()
            text = easyocr.Reader(['en'])
            text = text.readtext(imgRoi)
            # результат распознвания номера автомобиля
            # print('номер = ', text[0][1])

            # Сравнение номера с номерами из списка и замена автомобиля на фоновое изображение
            if text:
                if text[0][1] in del_number:
                    original_image[y1:y2, x1:x2] = img_backgr[y1:y2, x1:x2]
                    del_list.append(i)

# Удаление bounding boxes автомобилей из списка
boxes = [ele for idx, ele in enumerate(boxes) if idx not in del_list]

# Вывод исходной картинки
cv2.imshow('Result', image)
# вывод всех bounding boxes YOLO
plot_boxes(original_image, boxes, class_names, plot_labels=False)

cv2.waitKey(0)
cv2.destroyAllWindows()