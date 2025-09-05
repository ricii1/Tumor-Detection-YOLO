import cv2
import numpy as np

from ultralytics import YOLO
model = YOLO("best.pt")

# image ver
img = cv2.imread("pituitari2.jpg") 

results = model.predict(source=img, save=False, save_txt=False, conf=0.5, verbose=False)
for r in results:
    boxes = r.boxes

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    label = box.cls[0].item()  
    label_name = model.names[label]  
    (text_width, text_height), baseline = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (255, 0, 255), -1)
    cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()