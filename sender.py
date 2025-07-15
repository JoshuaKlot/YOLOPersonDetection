import cv2
import socket
import struct
import time

HOST = '0.0.0.0'
PORT = 9999 # cange the port to be different on each sender

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"[INFO] Waiting for connection on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print(f"[INFO] Connected to {addr}")
conn_file = conn.makefile('wb')

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Camera not available")
    conn_file.close()
    conn.close()
    server_socket.close()
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
            if not recording and not streaming:
                print("[INFO] Recording started. Press SPACE to stop and loop.")
                recording = True
                recorded_frames = []
            elif recording:
                print(f"[INFO] Recording stopped. {len(recorded_frames)} frames captured. Now looping.")
                recording = False
                streaming = True
                loop_index = 0
            elif streaming:
                print("[INFO] Looping stopped. Press SPACE to record a new segment.")
                streaming = False

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
            time.sleep(0.03)
            loop_index = (loop_index + 1) % len(recorded_frames)

    conn_file.write(struct.pack('<L', 0))  # End of stream

except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
    conn_file.close()
    conn.close()
    server_socket.close()