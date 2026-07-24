from ultralytics import YOLO
import cv2
import cvzone #lib used to draw the bounding box and labels of detected object

cap = cv2.VideoCapture(0)
model = YOLO("yolo26s.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            w,h = x2-x1, y2-y1
            conf = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            name = model.names[cls]
            cvzone.cornerRect(img, bbox=(x1, y1, w, h))
            cvzone.putTextRect(img, text=f"{name} {conf}", pos= (x1, y1-10), scale=1)
    cv2.imshow("Object Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break