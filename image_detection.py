from ultralytics import YOLO
import cv2

model = YOLO('YOLO_Model/yolov8m.pt')
# Resize the Window
# cv2.resizeWindow("win_name", 960, 540)
results = model("Images/cyc.jpg", show=True, line_thickness=1,  hide_conf=True)
cv2.waitKey(0)