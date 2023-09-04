# Author: Pari Malam

import cv2
import numpy as np
from colorama import init, Fore

init(autoreset=True)

XML_DIR = 'XML'
YOLO_DIR = 'YOLO'
WORDLIST = 'WORDLIST'

FACE_XML_PATH = f"{XML_DIR}/haarcascade_frontalface_default.xml"
HAND_XML_PATH = f"{XML_DIR}/haarcascade_hand.xml"
DEVICE_ID = 0
SCALE_FACTOR = 1.3
MIN_SIZE = (30, 30)

face_cascade = cv2.CascadeClassifier(FACE_XML_PATH)
hand_cascade = cv2.CascadeClassifier(HAND_XML_PATH)

net = cv2.dnn.readNet(f"{YOLO_DIR}/yolov3.weights", f"{YOLO_DIR}/yolov3.cfg")
output_layers = net.getUnconnectedOutLayersNames()

with open(f"{WORDLIST}/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

capture = cv2.VideoCapture(DEVICE_ID)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minSize=MIN_SIZE)

    return faces

def detect_hands(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minSize=MIN_SIZE)
    return hands

def detect_objects(frame):
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            detected_objects.append(label)

    return detected_objects

def main():
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow('Video', 800, 600)

    while True:
        ret, frame = capture.read()

        if not ret:
            print(Fore.GREEN + "[OpenCV] - " + Fore.RED + "Error: Unable to capture frame.")
            break

        faces = detect_faces(frame)
        hands = detect_hands(frame)
        objects = detect_objects(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if len(faces) > 0:
            print(Fore.GREEN + "[OpenCV] - " + Fore.WHITE + "Face(s) detected.")

        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if len(hands) > 0:
            print(Fore.GREEN + "[OpenCV] - " + Fore.CYAN + "Hand(s) detected.")

        if len(objects) > 0:
            print(Fore.GREEN + "[OpenCV] - " + Fore.YELLOW + "Objects detected: " + Fore.RED + ", ".join(objects))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
