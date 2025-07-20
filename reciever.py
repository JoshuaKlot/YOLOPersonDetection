import socket
import threading
import struct
import cv2
import numpy as np
import hashlib

REPEAT_THRESHOLD = 5

# Load YOLO ONNX model
#net = cv2.dnn.readNetFromONNX("yolov3-12.onnx")
#layer_names = net.getUnconnectedOutLayersNames()

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def compute_hash(data):
    return hashlib.md5(data).hexdigest()

# def apply_yolo(frame):
#     height, width = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     #net.setInput(blob)
#     #outputs = net.forward(layer_names)

#     #for output in outputs:
#     #    for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 label = f"{classes[class_id]} {confidence:.2f}"
#                 cv2.putText(frame, label, (x, y-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#     return frame

def handle_client(conn, addr, window_id):
    print(f"[INFO] Connected to sender at {addr}")
    conn_file = conn.makefile('rb')
    seen_hashes = set()
    repeat_count = 0
    try:
        while True:
            packed_len = conn_file.read(struct.calcsize('<L'))
            if not packed_len:
                break
            frame_size = struct.unpack('<L', packed_len)[0]
            if frame_size == 0:
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

            #frame = apply_yolo(frame)

            if repeat_count >= REPEAT_THRESHOLD:
                cv2.putText(frame, "Loop Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow(f"Stream from {addr[0]}:{addr[1]}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"[ERROR] {addr} - {e}")
    finally:
        conn_file.close()
        conn.close()
        cv2.destroyWindow(f"Stream from {addr[0]}:{addr[1]}")

# Main server socket setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9999))  # Listen on all interfaces
server_socket.listen(5)
print("[INFO] Receiver server started, waiting for senders...")

window_counter = 0

try:
    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr, window_counter), daemon=True).start()
        window_counter += 1
except KeyboardInterrupt:
    print("[INFO] Shutting down server.")
finally:
    server_socket.close()
    cv2.destroyAllWindows()
