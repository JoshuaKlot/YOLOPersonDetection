import cv2
import socket
import struct
import time
import random

RECEIVER_IP = '192.168.0.100'
RECEIVER_PORT = 9999

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
recorded_frames = []
loop_index = 0

inserting_loop = False
loop_insert_duration = 0
loop_insert_start_time = 0
next_loop_insert_time = time.time() + random.uniform(5, 15)  # initial delay before first loop

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Sender", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                print("[INFO] Recording started. Press SPACE to stop.")
                recording = True
                recorded_frames = []
            else:
                print(f"[INFO] Recording stopped. {len(recorded_frames)} frames captured.")
                recording = False
                loop_index = 0
                next_loop_insert_time = time.time() + random.uniform(5, 15)

        if key == ord('q'):
            break

        # Record frames while recording
        if recording:
            recorded_frames.append(frame.copy())

        now = time.time()

        # Decide whether to insert looped footage
        if recorded_frames and now >= next_loop_insert_time:
            inserting_loop = True
            loop_insert_duration = random.uniform(2, 5)  # seconds of loop to insert
            loop_insert_start_time = now
            loop_index = 0
            print(f"[INFO] Inserting looped footage for {loop_insert_duration:.1f} seconds.")

            # Schedule next loop insert
            next_loop_insert_time = now + random.uniform(10, 20)

        # Choose frame source
        if inserting_loop and recorded_frames:
            source_frame = recorded_frames[loop_index]
            loop_index = (loop_index + 1) % len(recorded_frames)

            if now - loop_insert_start_time > loop_insert_duration:
                inserting_loop = False
                print("[INFO] Returning to live feed.")
        else:
            source_frame = frame

        # Encode and send the frame
        result, encoded = cv2.imencode('.jpg', source_frame)
        frame_bytes = encoded.tobytes()
        conn_file.write(struct.pack('<L', len(frame_bytes)))
        conn_file.write(frame_bytes)
        conn_file.flush()

        time.sleep(0.03)  # Simulate ~30 FPS

    conn_file.write(struct.pack('<L', 0))  # End of stream

except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
    conn_file.close()
    client_socket.close()
