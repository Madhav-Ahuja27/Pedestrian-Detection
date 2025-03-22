import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

model = YOLO("yolov8n.pt")  
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(r"C:\Users\madha\Videos\pedestrians.mp4")
tracked_objects_data = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        if int(class_id) == 0 and score > 0.4:
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))

    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if track.is_confirmed():
            bbox = track.to_ltrb()
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, bbox)
            person_roi = frame[y1:y2, x1:x2]
            person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_roi_rgb)
            key_points = {}
            if results_pose.pose_landmarks:
                for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                    key_x = int(landmark.x * (x2 - x1)) + x1
                    key_y = int(landmark.y * (y2 - y1)) + y1
                    key_points[idx] = (key_x, key_y)

            tracked_objects_data[track_id] = {
                "bbox": (x1, y1, x2, y2),
                "key_points": key_points
            }

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for _, (kx, ky) in key_points.items():
                cv2.circle(frame, (kx, ky), 3, (0, 0, 255), -1)

            print(f"ID {track_id}: BBox = {x1, y1, x2, y2}, Key Points = {key_points}")

    cv2.imshow("DeepSORT Tracking with Keypoints", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
