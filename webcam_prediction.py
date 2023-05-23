import cv2
import math
from ultralytics import YOLO

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("Videos/people.mp4")

model = YOLO('YOLO_Model/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog",
              "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop",
              "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book",
              "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)   
    for bound_box in results:
        boxes = bound_box.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print("x1, y1, x2, y2", x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            conf = math.ceil((box.conf[0]*100))/100
            class_name = int(box.cls[0])
            cv2.putText(img, f'{classNames[class_name]}', (x1, y1), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0, color = (125, 246, 55), thickness=1)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' key is pressed
        break