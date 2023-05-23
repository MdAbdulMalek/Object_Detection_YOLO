import cv2
import math
from ultralytics import YOLO
from sort import *

cap = cv2.VideoCapture("Videos/cars.mp4")

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

mask = cv2.imread("Images/mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [180, 397, 683, 397]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)  

    detections = np.empty((0,5))

    for bound_box in results:
        boxes = bound_box.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print("x1, y1, x2, y2", x1, y1, x2, y2)
            
            conf = math.ceil((box.conf[0]*100))/100
            class_id = int(box.cls[0])
            class_name = classNames[class_id]
            if class_name == "car" and conf > 0.3:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                # cv2.putText(img, f'{class_name}', (x1, y1), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0, 
                #             color = (125, 246, 55), thickness=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
    total_count = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for total in total_count:
        x1, y1, x2, y2, id = total
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cv2.putText(img, f'{int(id)}',(x1, y1),cv2.FONT_HERSHEY_PLAIN,5,(50,80,255),5)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)


    cv2.putText(img,str(len(totalCount)),(200,85),cv2.FONT_HERSHEY_PLAIN,5,(50,80,255),8)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' key is pressed
        break