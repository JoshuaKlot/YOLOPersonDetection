import socket
import threading
import struct
import cv2
import numpy as np
import hashlib

# Threshold for detecting repeated frames (i.e., looped video)
REPEAT_THRESHOLD = 5

# Load YOLO classes from coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Computes MD5 hash of frame data to detect duplicates
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

# Handles a single incoming connection (i.e., one camera stream)
def handle_client(conn, addr, window_id):
    print(f"[INFO] Connected to sender at {addr}")
    
    # Wrap socket in a file-like object for easier binary reading
    conn_file = conn.makefile('rb')
    
    # Each client (camera stream) gets its own set of seen hashes
    seen_hashes = set()
    repeat_count = 0

    try:
        while True:
            # Read the size of the incoming frame (4 bytes, little-endian unsigned long)
            packed_len = conn_file.read(struct.calcsize('<L'))
            if not packed_len:
                break  # Connection closed

            frame_size = struct.unpack('<L', packed_len)[0]
            if frame_size == 0:
                break  # Sender indicates end of stream

            # Read the actual frame data
            frame_data = conn_file.read(frame_size)
            frame_hash = compute_hash(frame_data)

            # Check if this frame is a repeat (based on hash)
            if frame_hash in seen_hashes:
                repeat_count += 1
            else:
                seen_hashes.add(frame_hash)
                repeat_count = 0

            # Decode image data to OpenCV frame
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue  # Skip invalid frames

            # Optional: Apply object detection (currently commented out)
            # frame = apply_yolo(frame)

            # Overlay "Loop Detected" warning if repeated frames are seen
            if repeat_count >= REPEAT_THRESHOLD:
                cv2.putText(frame, "Loop Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Display the video frame in a window named by the client IP and port
            cv2.imshow(f"Stream from {addr[0]}:{addr[1]}", frame)

            # Exit display if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] {addr} - {e}")
    finally:
        # Clean up on disconnect or error
        conn_file.close()
        conn.close()
        cv2.destroyWindow(f"Stream from {addr[0]}:{addr[1]}")

# --- Main server setup ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9999))  # Listen on all network interfaces
server_socket.listen(5)  # Allow up to 5 pending connections
print("[INFO] Receiver server started, waiting for senders...")

window_counter = 0

try:
    while True:
        # Accept a new client (camera stream)
        conn, addr = server_socket.accept()
        
        # Start a new thread for each client connection
        threading.Thread(target=handle_client, args=(conn, addr, window_counter), daemon=True).start()
        window_counter += 1
except KeyboardInterrupt:
    print("[INFO] Shutting down server.")
finally:
    # Close server socket and all OpenCV windows on exit
    server_socket.close()
    cv2.destroyAllWindows()
