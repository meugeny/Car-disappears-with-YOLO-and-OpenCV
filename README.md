# Машина исчезает
Задачей проекта является удаление из фото камеры видеонаблюдения автомобилей с заданными номерами. 
Исходными данными является фоновое изображение без машин, изображение с машинами, список номеров автомобилей для скрытия.  
В папке `images` представлены три пары таких изображений.  
<img src="/images/parking.jpeg" width="200"/> <img src="/images/img_backgr.jpeg" width="200"/>    
<img src="/images/parking2.jpeg" width="200"/> <img src="/images/img_backgr2.jpeg" width="200"/>  
<img src="/images/parking3.jpeg" width="200"/> <img src="/images/img_backgr3.jpeg" width="200"/>  
У данной программы может быть множество применений: 
* для публичных камер не показывать неразглашемых людей
* создавать списки автомобилей, которые не используют общие КПП

В качестве развития возможно перенести модель на распознавание видео с камер видеонаблюдения и для точного вырезания автомобилей использовать попиксельную сегментацию.  

Детекция автомобилей осуществляется One-shot моделью YOLO натренированную на датасете COCO. Веса модели `https://pjreddie.com/media/files/yolov3.weights`.    
<img src="/images/ex_yolo_03.png" width="600"/>  

Для поиска российских номеров на изображении используется обученный классификатор `haarcascade_russian_plate_number.xml`. Классификатор формируется на примитивах Хаара.  
<img src="/images/ex_number.png" width="300"/>  

Распознавание производится моделью EaseOCR.

# Результат работы модели:  
<img src="/images/parking.jpeg" width="400"/> <img src="/images/ex_yolo.png" width="400"/>  
<img src="/images/parking2.jpeg" width="400"/> <img src="/images/ex_yolo_02.png" width="400"/>  
<img src="/images/ex_yolo_01.png" width="400"/> <img src="/images/ex_yolo_021.png" width="400"/>

