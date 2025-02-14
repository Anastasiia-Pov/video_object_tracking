from ultralytics import YOLO
import cv2
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFont

def process_box_coords(x0, y0, x1, y1, rows, cols):
    # превращаем строки в числа и фиксируем координаты рамки, чтобы они не выходили за пределы кадра
    x0 = np.clip(int(x0), 0, cols)
    x1 = np.clip(int(x1), 0, cols)
    y0 = np.clip(int(y0), 0, rows)
    y1 = np.clip(int(y1), 0, rows)

    # чтобы у нас ширина и высота рамки была не отрицательной, 
    # переставляем местами нулевую и первую координаты, если первая больше нулевой
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    return x0, y0, x1, y1

def draw_bbox_with_text(
    image:np.array,
    bbox_coords:tuple,
    bbox_width:int,
    class_name:str,
    color:tuple,
    font:ImageFont.FreeTypeFont,
    ):
    '''
    image:np.array - массив пикселей изображения
    bbox_coords:tuple|list, - координаты рамки в формате x0,y0,x1,y1
    bbox_width:int, - ширина рамки
    class_name:str, - имя выводимого класса
    color:tuple|list, - цвет рамки
    font:ImageFont.FreeTypeFont, - шрифт текста
    '''

    x0, y0, x1, y1 = bbox_coords
    image =  Image.fromarray(image)
    cols, rows = image.size
    #x0, y0, x1, y1 = process_box_coords(x0, y0, x1, y1, rows, cols)

    draw = ImageDraw.Draw(image)

    # рисуем прямоугольник для общей рамки...
    draw.rectangle(bbox_coords, outline=color, width=bbox_width)
   

    # определяем цвет шрифта исходя из яркости ЧБ эквивалента цвета класса
    r, g, b = color
    grayscale = int(0.299*r + 0.587*g + 0.114*b)
    # пороговая фильтрация работает на удивление хорошо...
    font_color = 255 if grayscale < 128 else 0

    # вычисляем координаты текста - посередине рамки
    text_coords = ((x1+x0)//2, (y1+y0)//2)

    font_size = font.size

    # квадратный корень почему-то работает очень хорошо для вычисления ширины рамки текста...
    text_bbow_width = np.round(np.sqrt(font_size)).astype(int)
    # вычисляем зазор между рамкой текста и текстом
    text_bbox_spacing = text_bbow_width//3 if text_bbow_width//3 > 1 else 1

    # определяем координаты обрамляющего текст прямоугольника
    text_bbox = draw.textbbox(text_coords, class_name, font=font, anchor='mm') # anchor='mm' означает расположение текста посередине относительно координат
    # расширяем рамку на 3 пикселя в каждом направлении
    text_bbox = tuple(np.add(text_bbox, (-text_bbow_width, -text_bbow_width, text_bbow_width, text_bbow_width)))

    # рисуем прямоугольник для текста
    draw.rectangle(text_bbox, outline=(font_color, font_color, font_color), fill=color, width=bbox_width-text_bbox_spacing)
    # пишем текст
    draw.text(text_coords, class_name, font=font, anchor='mm', fill=(font_color, font_color, font_color))

    return np.array(image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', help='Choose YOLO model from the list yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov9t, yolov8s, yolov9m, yolov9c, yolov9e, yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x, yolo11n, yolo11s, yolo11m, yolo11l, yolo11x')
    parser.add_argument('--class_to_track', help='Choose class to track from the list of MS COCO dataset (see https://blog.roboflow.com/microsoft-coco-classes/)')
    parser.add_argument('--displaying_font', default='FiraCode-SemiBold.ttf', help='Path to *.ttf file with the font of bboxes class name')
    parser.add_argument('--video_source', help='0, if the source is web-cam, or absolute path to video')

    sample_args = [
        '--yolo_model', 'yolov8n',
        '--class_to_track', 'person',
        '--video_source', r'c:\Users\admin\python_programming\DATA\AVABOS\DATASET_V0\physical\video\c-0_v-0-0_F-0-0_10.0-12.0_NOAGGR.mp4']
    args = parser.parse_args(sample_args)
  
    yolo_model = args.yolo_model
    class_to_track = args.class_to_track
    video_source = args.video_source
    path_to_font = args.displaying_font

    cap = cv2.VideoCapture(video_source)

    # Load the YOLO model
    model = YOLO(yolo_model)

    cv2.namedWindow("YOLO Tracking")
    
    # Loop through the video frames
    while cap.isOpened():
        
        key = cv2.waitKey(20) 
        
        # следующий кадр читается по нажатию клавиши "пробел" на клавиатуре
        if key & 0xFF == 32:
            # Read a frame from the video
            success, frame = cap.read()
            if not success:                                     
                break

            rows, cols, channels = frame.shape
            print(frame)

            # определяем размер шрифта исходя из размера изображения
            font_size = min(rows,cols)//30

            # вычисляем ширину рамки. Квадратный корень работает хорошо...
            line_width = np.round(np.sqrt(font_size).astype(int))
            # устанавливаем шрифт для указания размечаемых людей
            font = ImageFont.truetype("FiraCode-SemiBold.ttf", font_size)

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # выполнение трекинга 
            results = model.track(frame, persist=True)[0]

            # получение координат рамок
            bboxes = results.boxes.xyxy.long().numpy()
            # получение индексов объектов
            ids = results.boxes.id.long().numpy()
            
            # получение списка детектированных классов
            detected_classes = [model.names[cls_idx] for cls_idx in results.boxes.cls.long().numpy()]
            
            # строки и столбцы изображения нужны для создания объектов рамок
            img_rows, img_cols = results[0].orig_img.shape[:-1]
            
            
            for bbox, id, class_name in zip(bboxes, ids, detected_classes):
                if class_name == class_to_track:
                    x0,y0,x1,y1 = bbox
                    #x0,y0,x1,y1 = x0 - bbox_append_value, y0 - bbox_append_value, x1 + bbox_append_value, y1 + bbox_append_value
                    # делаем так, чтобы рамка не выходила за пределы кадра
                    x0,y0,x1,y1 = process_box_coords(x0,y0,x1,y1, img_rows, img_cols)
                    # пока что оставляем всего лишь один цвет - черный
                    color = (0,0,0)
                    displaying_name = f'{class_name},{id}'
                    # рисуем рамку
                    frame = draw_bbox_with_text(frame, (x0,y0,x1,y1), line_width, displaying_name, color, font)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()