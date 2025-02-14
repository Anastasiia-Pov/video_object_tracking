from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO('yolov8n-pose.pt')

    # Open the video file
    #video_path = r"C:\Users\admin\python_programming\DATA\AVABOS\test_bboxes\4LUoqxnyxlE(+)_._0.066-40.066.mp4"
    video_path = r'I:\AVABOS\4LUoqxnyxlE(+)+ - test\4LUoqxnyxlE(+).mp4'
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow("YOLOv8 Tracking")
    success, annotated_frame = cap.read()

    # Loop through the video frames
    while cap.isOpened():
        # Display the annotated frame
        cv2.imshow("YOLOv8 segmentation", annotated_frame)
        # Read a frame from the video
        key = cv2.waitKey(20)
        #print(key & 0xFF)
        if key & 0xFF == 32:
            success, frame = cap.read()
            if not success:                                     
                break

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()