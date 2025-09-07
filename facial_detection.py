import cv2
import socket
import struct
import time
import numpy as np

# Ask the user for the receiver IP
RECEIVER_IP = input("Enter the receiver IP address: ").strip()
RECEIVER_PORT = 9999
print(f"[INFO] Connecting to receiver at {RECEIVER_IP}:{RECEIVER_PORT}...")

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((RECEIVER_IP, RECEIVER_PORT))
conn_file = client_socket.makefile('wb')

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cam.isOpened():
    print("[ERROR] Camera not available")
    conn_file.close()
    client_socket.close()
    exit()

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("[INFO] Controls:")
print("  SPACE - Start/stop recording")
print("  L - Play recorded loop manually")
print("  T - Train face recognition on current face")
print("  A - Toggle auto-loop when target face detected")
print("  Q - Quit")

# Variables
recording = False
recorded_frames = []
loop_index = 0
inserting_loop = False
target_face_trained = False
auto_loop_enabled = False
face_detected_counter = 0
FACE_DETECTION_THRESHOLD = 5  # Frames to confirm face detection

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles around faces and check for target face
        target_face_detected = False
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # If we have a trained face, try to recognize it
            if target_face_trained:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (100, 100))
                
                label, confidence = face_recognizer.predict(roi_gray)
                
                # Lower confidence means better match (threshold around 50-100 works well)
                if confidence < 80:  
                    cv2.putText(frame, f"Target Face (conf: {confidence:.1f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    target_face_detected = True
                else:
                    cv2.putText(frame, f"Unknown (conf: {confidence:.1f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Handle automatic loop triggering
        if auto_loop_enabled and target_face_detected and recorded_frames:
            face_detected_counter += 1
            if face_detected_counter >= FACE_DETECTION_THRESHOLD and not inserting_loop:
                print("[INFO] Target face detected! Starting automatic loop playback...")
                inserting_loop = True
                loop_index = 0
                face_detected_counter = 0
        else:
            face_detected_counter = 0
        
        # Display status
        status_text = []
        if recording:
            status_text.append("REC")
        if target_face_trained:
            status_text.append("FACE TRAINED")
        if auto_loop_enabled:
            status_text.append("AUTO-LOOP ON")
        if inserting_loop:
            status_text.append("LOOPING")
            
        if status_text:
            cv2.putText(frame, " | ".join(status_text), (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("Sender", frame)
        
        # Handle keyboard input
        if key == ord(' '):
            # Start/Stop recording
            if not recording:
                print("[INFO] Recording started. Press SPACE to stop.")
                recording = True
                recorded_frames.clear()
            else:
                print(f"[INFO] Recording stopped. {len(recorded_frames)} frames captured.")
                recording = False
                loop_index = 0
                
        elif key == ord('l') and recorded_frames:
            # Manual loop playback
            print("[INFO] Playing back recorded footage...")
            inserting_loop = True
            loop_index = 0
            
        elif key == ord('t') and len(faces)>0:
            # Train face recognition on current face
            print("[INFO] Training face recognition on detected face...")
            # Use the first detected face for training
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (100, 100))
            
            # Create training data (you might want to collect multiple samples)
            training_faces = [roi_gray]
            training_labels = [1]  # Label 1 for target face
            
            face_recognizer.train(training_faces, np.array(training_labels))
            target_face_trained = True
            print("[INFO] Face training completed!")
            
        elif key == ord('a'):
            # Toggle auto-loop
            auto_loop_enabled = not auto_loop_enabled
            print(f"[INFO] Auto-loop {'enabled' if auto_loop_enabled else 'disabled'}")
            
        elif key == ord('q'):
            break
        
        # Record live frames if recording is on
        if recording:
            recorded_frames.append(frame.copy())
        
        # Decide which frame to send
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
    
    # Send end of stream signal
    conn_file.write(struct.pack('<L', 0))
    
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
    conn_file.close()
    client_socket.close()
