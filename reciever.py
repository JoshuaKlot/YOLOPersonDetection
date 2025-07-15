import cv2
import socket
import struct
import threading
import numpy as np
import hashlib

SENDER_LIST = [
    ('127.0.0.1', 9999)
    # Add more (IP, PORT) tuples as needed
]

REPEAT_THRESHOLD = 5

# Load YOLOv3 or v4 model
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def compute_hash(data):
    return hashlib.md5(data).hexdigest()

def apply_yolo(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    for output in outputs:
        for detection in output:
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def handle_stream(ip, port, window_name):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        conn_file = sock.makefile('rb')

        print(f"[INFO] Connected to {ip}:{port}")
        seen_hashes = set()
        repeat_count = 0

        while True:
            packed_len = conn_file.read(struct.calcsize('<L'))
            if not packed_len:
                break
            frame_size = struct.unpack('<L', packed_len)[0]
            if frame_size == 0:
                print(f"[INFO] End of stream from {ip}:{port}")
                break

            frame_data = conn_file.read(frame_size)
            frame_hash = compute_hash(frame_data)

            if frame_hash in seen_hashes:
                repeat_count += 1
            else:
                seen_hashes.add(frame_hash)
                repeat_count = 0

            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame = apply_yolo(frame)

            if repeat_count >= REPEAT_THRESHOLD:
                cv2.putText(frame, "Loop Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        conn_file.close()
        sock.close()
        cv2.destroyWindow(window_name)

    except Exception as e:
        print(f"[ERROR] Stream {ip}:{port} - {e}")

# Start threads for each sender
threads = []
for idx, (ip, port) in enumerate(SENDER_LIST):
    window_name = f"Stream {idx+1} - {ip}:{port}"
    t = threading.Thread(target=handle_stream, args=(ip, port, window_name), daemon=True)
    threads.append(t)
    t.start()

# Keep main thread alive until all windows are closed
try:
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()
