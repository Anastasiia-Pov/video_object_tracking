# video_object_tracking

Учебная программа, которая читает видео из файла или с вэб-камеры и выполняет трэкинг объектов заданных классов. 
При этом осуществлен функционал аргументов командной строки (argparse.ArgumentParser) в файле [test_tracker.py](https://github.com/Anastasiia-Pov/video_object_tracking/blob/main/test_tracker.py):
- '--yolo_model' - выбор модели YOLO (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov9t, yolov8s, yolov9m, yolov9c, yolov9e, yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x, yolo11n, yolo11s, yolo11m, yolo11l, yolo11x);
- '--class_to_track' - выбор объекта для отслеживания (со списком всех классов можно ознакомиться на сайте https://blog.roboflow.com/microsoft-coco-classes/);
- '--displaying_font' - путь к файлу *.ttf со шрифтом класса bboxes;
- '--video_source' - 0, если веб-камера, или абсолютный путь до видео