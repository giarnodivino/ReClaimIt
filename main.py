import cv2

from ultralytics import YOLO
import supervision as sv

points_dict = {
    "cell phone": 2000,
    "laptop": 2500,
    "chair": 1500,
    "tv": 2000,
    "bottle": 500,
    "keyboard": 500,
    "backpack": 300,
    "mouse": 500,

}

stream = cv2.VideoCapture(0)
stream.set(3, 1280)
stream.set(4, 720)



if not stream.isOpened():
    print("No stream")
    exit()

fps = stream.get(cv2.CAP_PROP_FPS)
width = int(stream.get(3))
height = int(stream.get(4))

model = YOLO("yolo11s.pt")


box_annotator = sv.BoxAnnotator(
    thickness=2,
)
label_annotator = sv.LabelAnnotator()

while(True):
    ret, frame = stream.read()
    if not ret:
        print("No more stream")
        break

    frame = cv2.resize(frame, (width, height))

    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels_with_points = []
    for class_id in detections.class_id:
        class_name = model.model.names[class_id]
        points = points_dict.get(class_name.lower(), 0)
        label_text = f"{class_name}, {points} points"
        labels_with_points.append(label_text)


    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels_with_points)


    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        print("Stream Ended")
        break

stream.release()
cv2.destroyAllWindows()