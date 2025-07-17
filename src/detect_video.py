from ultralytics import YOLO
import cv2
from datetime import datetime



stream = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path or else for accesing certain video use path in brackets

if not stream.isOpened():
    print("hell nha bud they are not giving access :/")
    exit()

width = int(stream.get(3))
height = int(stream.get(4))  # Get the width and height of the video stream
fps = stream.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video stream
filename = f"data/outputs/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
output = cv2.VideoWriter("data/outputs/intrusion_output.mp4",cv2.VideoWriter_fourcc('m','p','4','v'),fps=fps, frameSize=(width, height)) #be specific and also choose 4cc code according to ur videosaved

model = YOLO("yolov8n.pt")  # Load the YOLO model (change to your model path if needed)

while True:
    ret, frame = stream.read()
    if not ret:
        print("NO more stream buddy")
        break

    results = model(frame)

    # Process the results (e.g., draw bounding boxes)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frame = cv2.resize(frame, (width, height))  # Resize the frame to 640x480
    output.write(frame)  # Write the frame to the output video file
    cv2.imshow("Webcam!", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
stream.release()
cv2.destroyAllWindows() #! Close all OpenCV windowsprint

