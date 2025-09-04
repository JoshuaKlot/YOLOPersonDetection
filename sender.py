import cv2
import socket
import struct
import time

# Ask the user for the receiver IP
RECEIVER_IP = input("Enter the receiver IP address: ").strip()
RECEIVER_PORT = 9999

print(f"[INFO] Connecting to receiver at {RECEIVER_IP}:{RECEIVER_PORT}...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((RECEIVER_IP, RECEIVER_PORT))
conn_file = client_socket.makefile('wb')

# ---- Find the first available camera ----
def find_camera(max_index=10):
    for i in range(max_index):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"[INFO] Using camera index {i}")
            return cam
        cam.release()
    return None

cam = find_camera()
if cam is None:
    print("[ERROR] No available camera found.")
    conn_file.close()
    client_socket.close()
    exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Ready. Press SPACE to start/stop recording, L to play loop, Q to quit.")

recording = False
recorded_frames = []
loop_index = 0
inserting_loop = False

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("Sender", frame)

        # Start/Stop recording
        if key == ord(' '):
            if not recording:
                print("[INFO] Recording started. Press SPACE to stop.")
                recording = True
                recorded_frames.clear()  # Delete previous recording
            else:
                print(f"[INFO] Recording stopped. {len(recorded_frames)} frames captured.")
                recording = False
                loop_index = 0

        # Play the recorded loop
        if key == ord('l') and recorded_frames:
            print("[INFO] Playing back recorded footage...")
            inserting_loop = True
            loop_index = 0

        # Quit
        if key == ord('q'):
            break

        # Record live frames if recording is on
        if recording:
            recorded_frames.append(frame.copy())

        # Decide frame to send
        if inserting_loop and recorded_frames:
            source_frame = recorded_frames[loop_index]
            loop_index += 1

            if loop_index >= len(recorded_frames):
                inserting_loop = False
                print("[INFO] Loop playback finished. Returning to live feed.")
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
