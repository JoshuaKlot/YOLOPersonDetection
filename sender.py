import cv2
import socket
import struct
import time

RECEIVER_IP = 'localhost'  # Change to your receiver's IP address
RECEIVER_PORT = 9999       # Must match receiver's listening port

print(f"[INFO] Connecting to receiver at {RECEIVER_IP}:{RECEIVER_PORT}...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((RECEIVER_IP, RECEIVER_PORT))
conn_file = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Camera not available")
    conn_file.close()
    client_socket.close()
    exit()

print("[INFO] Ready. Press SPACE to start/stop recording, Q to quit.")

recording = False
streaming = False
recorded_frames = []
loop_index = 0

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Sender", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                print("[INFO] Recording started. Press SPACE to stop and loop.")
                recording = True
                streaming = False
                recorded_frames = []
            elif recording:
                print(f"[INFO] Recording stopped. {len(recorded_frames)} frames captured. Now looping. Press SPACE to make new loop.")
                recording = False
                streaming = True
                loop_index = 0

        if key == ord('q'):
            break

        if recording:
            recorded_frames.append(frame.copy())

        if streaming and recorded_frames:
            result, encoded = cv2.imencode('.jpg', recorded_frames[loop_index])
            frame_bytes = encoded.tobytes()
            conn_file.write(struct.pack('<L', len(frame_bytes)))
            conn_file.write(frame_bytes)
            conn_file.flush()
            time.sleep(0.03)  # Simulate ~30 FPS
            loop_index = (loop_index + 1) % len(recorded_frames)

    conn_file.write(struct.pack('<L', 0))  # End of stream

except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
    conn_file.close()
    client_socket.close()
