import cv2 
from cvzone.FaceDetectionModule import FaceDetector

detector = FaceDetector()

cap=cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img, bboxes = detector.findFaces(img)
    if bboxes:
        center = bboxes[0]['center']
    cv2.imshow("Object Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break